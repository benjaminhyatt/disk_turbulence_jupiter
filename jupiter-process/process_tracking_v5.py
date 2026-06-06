"""
Usage:
    process_tracking.py <file> [options]

Options:
    --output=<str>          prefix for output file [default: processed_tracking]
    --t_out_start=<float>   sim time to begin tracking [default: 149.]
    --t_out_end=<float>     sim time to stop tracking [default: 251.]
    --t_hist_start=<float>  sim time to start including tracking results in distributions [default: 149.]
    --t_hist_end=<float>    sim time to stop including tracking results in distributions [default: 251.]

    --use_cutoff=<bool>     ignore grid data beyond a specified radius [default: True]
    --r_cutoff=<float>      cutoff radius; if None, computed from Lgamma [default: None]
    --use_stddev=<bool>     ignore grid data below a multiple of the standard deviation [default: False]
    --bin_width_phi=<int>   dedalus grid points per bin in phi [default: 1]
    --bin_width_r=<int>     dedalus grid points per bin in r [default: 1]

    --use_interp=<bool>     use bivariate spline to refine extremum location [default: True]
    --use_optimize=<bool>   use optimization rather than sampling to find spline extremum [default: True]
    --local_size_phi=<int>  phi grid points passed to spline (>= 2) [default: 2]
    --local_size_r=<int>    r grid points passed to spline (>= 2, recommend >= 3) [default: 3]
    --precision_phi=<int>   even number of sample points per phi interval for spline [default: 2]
    --precision_r=<int>     even number of sample points per r interval for spline [default: 4]

    --max_jump_r=<float>    max allowed jump in r between frames before flagging as glitch [default: 0.1]
    --max_jump_phi=<float>  max allowed jump in phi (rad) between frames before flagging as glitch [default: 1.5]
    --jump_vort_fac=<float>  if jump detected, only accept if new vorticity exceeds old by this factor [default: 2.0]

    --extract_vel=<bool>    extract CPC velocity profile at each write [default: True]
    --n_r_profile=<int>     number of radial points to sample for velocity profile [default: 384]

    --subtract_rw=<bool>    subtract dominant RW from velocity before CPC transform [default: True]
    --proj_file=<str>       path to processed projection FFT .npy file [default: None]
    --evp_file=<str>        path to processed EVP .npy file [default: None]
    --evp_mode_idx=<int>    index of dominant EVP mode (sort_im_inc order) [default: 1]
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
from scipy.interpolate import RectSphereBivariateSpline as splinefit
from scipy.stats import norm, rice
from scipy.optimize import minimize

print("args read in")
print(args)

file_str      = args['<file>']
output_prefix = args['--output']

t_out_start  = float(args['--t_out_start'])
t_out_end    = float(args['--t_out_end'])
t_hist_start = float(args['--t_hist_start'])
t_hist_end   = float(args['--t_hist_end'])

use_cutoff     = eval(args['--use_cutoff'])
use_stddev     = eval(args['--use_stddev'])
r_cutoff_given = args['--r_cutoff'] != 'None'
if r_cutoff_given:
    r_cutoff = float(args['--r_cutoff'])

bin_width_phi = int(args['--bin_width_phi'])
bin_width_r   = int(args['--bin_width_r'])

use_interp = eval(args['--use_interp'])
if use_interp:
    use_optimize   = eval(args['--use_optimize'])
    local_size_phi = int(args['--local_size_phi'])
    local_size_r   = int(args['--local_size_r'])
    if local_size_phi < 2 or local_size_r < 2:
        raise ValueError("local_size_phi and local_size_r must each be >= 2")
    if local_size_r < 3:
        print("warning: spline may fail near pole; recommend local_size_r >= 3")
    precision_phi = int(args['--precision_phi'])
    precision_r   = int(args['--precision_r'])
    if precision_phi % 2 != 0 or precision_r % 2 != 0:
        raise ValueError("precision_phi and precision_r must be even integers")

max_jump_r    = float(args['--max_jump_r'])
max_jump_phi  = float(args['--max_jump_phi'])
jump_vort_fac = float(args['--jump_vort_fac'])

extract_vel  = eval(args['--extract_vel'])
n_r_profile  = int(args['--n_r_profile'])

subtract_rw  = eval(args['--subtract_rw'])
proj_file    = args['--proj_file']
evp_file     = args['--evp_file']
evp_mode_idx = int(args['--evp_mode_idx'])
if subtract_rw and (proj_file == 'None' or evp_file == 'None'):
    raise ValueError("--subtract_rw=True requires --proj_file and --evp_file to be specified")

### string parsing ###
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec / 10) * 10 ** (sgn * exp)

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]
Nphi      = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr        = int(output_suffix.split('Nr_')[1].split('_')[0])
alpha_read = str_to_float(output_suffix.split('alpha_')[1].split('_')[0])
gamma_read = str_to_float(output_suffix.split('gam_')[1].split('_')[0])
eps_read   = str_to_float(output_suffix.split('eps_')[1].split('_')[0])
nu_read    = str_to_float(output_suffix.split('nu_')[1].split('_')[0])

alpha_vals = np.array((1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
eps_vals   = np.array([3.3e-1, 1.0, 2.0])
nu_vals    = np.array([5e-5, 2e-4])
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
eps   = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu    = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]

if Nphi % bin_width_phi != 0:
    print("Warning: bin_width_phi does not divide Nphi evenly -- bins will be unequal.")

### coordinate conversion ###
Rfactor = 3
def r_to_th(r_in):
    return np.arcsin(r_in / Rfactor)
def th_to_r(th_in):
    return Rfactor * np.sin(th_in)

### Dedalus setup ###
dealias = 3/2
dtype   = np.float64
coords  = d3.PolarCoordinates('phi', 'r')
dist    = d3.Distributor(coords, dtype=dtype)
disk    = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)

phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
vort  = dist.Field(name='vort', bases=disk)
u_field = dist.VectorField(coords, name='u', bases=disk)

phi_1d = phi_deal[:, 0]   # shape (Nphi_deal,)
r_1d   = r_deal[0, :]     # shape (Nr_deal,)

# meshes with shape (Nphi_deal, Nr_deal)
phi_mesh   = np.tile(phi_1d[:, np.newaxis], (1, len(r_1d)))
theta_1d   = r_to_th(r_1d)
theta_mesh = np.tile(theta_1d[np.newaxis, :], (len(phi_1d), 1))
r_mesh     = np.tile(r_1d[np.newaxis, :],     (len(phi_1d), 1))

### helper functions for spline meshing ###
def choose_mesh(lon_mesh, lat_mesh, data, lon_idx, lat_idx, Nlon, Nlat, size_lon, size_lat):
    bounds = {}
    lat_idx_inner = max(0, lat_idx - size_lat)
    lat_idx_outer = min(lat_idx + size_lat, Nlat - 1)
    lmc1 = lon_mesh[:, lat_idx_inner:lat_idx_outer+1]
    lac1 = lat_mesh[:, lat_idx_inner:lat_idx_outer+1]
    dc1  = data[:,   lat_idx_inner:lat_idx_outer+1]
    bounds['lat_idxs']       = [lat_idx_inner, lat_idx_outer]
    bounds['lat_sub_mesh_g'] = lac1[0, :]
    bounds['lat_pole_flag']  = lat_idx - size_lat < 0
    lon_cut = lat_idx_inner != 0

    if lon_cut and (lon_idx - size_lon < 0):
        wa, ea = 0, lon_idx + size_lon
        wb, eb = Nlon + (lon_idx - size_lon), Nlon - 1
        lm = np.vstack((lmc1[wb:eb+1,:], lmc1[wa:ea+1,:]))
        la = np.vstack((lac1[wb:eb+1,:], lac1[wa:ea+1,:]))
        ds = np.vstack((dc1[wb:eb+1,:],  dc1[wa:ea+1,:]))
        bounds.update({'lon_idxs': None, 'lon_std_flag': None,
                       'lon_sub_mesh_g': lm[:,0],
                       'lon_a_idxs': [wa,ea], 'lon_b_idxs': [wb,eb],
                       'lon_ab_flag': True,
                       'lon_a_bds': [lmc1[wa,0], lmc1[ea,0]],
                       'lon_b_bds': [lmc1[wb,0], 2*np.pi]})
    elif lon_cut and (lon_idx + size_lon > Nlon - 1):
        wa, ea = lon_idx - size_lon, Nlon - 1
        wb, eb = 0, (lon_idx + size_lon) - Nlon
        lm = np.vstack((lmc1[wa:ea+1,:], lmc1[wb:eb+1,:]))
        la = np.vstack((lac1[wa:ea+1,:], lac1[wb:eb+1,:]))
        ds = np.vstack((dc1[wa:ea+1,:],  dc1[wb:eb+1,:]))
        bounds.update({'lon_idxs': None, 'lon_std_flag': None,
                       'lon_a_idxs': [wa,ea], 'lon_b_idxs': [wb,eb],
                       'lon_ab_flag': False,
                       'lon_a_bds': [lmc1[wa,0], 2*np.pi],
                       'lon_b_bds': [lmc1[wb,0], lmc1[eb,0]]})
    elif lon_cut:
        w, e = lon_idx - size_lon, lon_idx + size_lon
        lm = lmc1[w:e+1,:]
        la = lac1[w:e+1,:]
        ds = dc1[w:e+1,:]
        bounds.update({'lon_idxs': [w,e], 'lon_std_flag': True,
                       'lon_bds': [lm[0,0], lm[-1,0]],
                       'lon_a_idxs': None, 'lon_b_idxs': None,
                       'lon_ab_flag': None, 'lon_a_bds': None, 'lon_b_bds': None})
    else:
        lm, la, ds = lmc1, lac1, dc1
        bounds.update({'lon_idxs': [0, Nlon-1], 'lon_std_flag': False,
                       'lon_bds': [0, 2*np.pi],
                       'lon_a_idxs': None, 'lon_b_idxs': None,
                       'lon_ab_flag': None, 'lon_a_bds': None, 'lon_b_bds': None})
    return lm, la, ds, bounds

def lat_test(lat_g, lat_idxs, prec, near_pole):
    inner, outer = lat_idxs
    pts = np.linspace(0, lat_g[0], prec+1, endpoint=False)[1:] if near_pole else np.array([])
    for i in range(outer - inner):
        pts = np.concatenate((pts, np.linspace(lat_g[i], lat_g[i+1], prec+1, endpoint=False)))
    return np.concatenate((pts, [lat_g[outer - inner]]))

def _unwrap(lons):
    lons = lons.copy()
    lons[lons >= np.pi] -= 2*np.pi
    return lons[np.argsort(lons)]

def lon_test_std(bds, idxs, prec, std_flag):
    w, e = idxs
    N = int(np.round((prec+1)*(e - w + int(not std_flag))) + int(std_flag))
    return np.linspace(bds[0], bds[-1], N, endpoint=std_flag)

def lon_test_ab(a_bds, b_bds, a_idxs, b_idxs, prec, ab_flag):
    wa, ea = a_idxs; wb, eb = b_idxs
    Na = int(np.round((prec+1)*(ea-wa+int(not ab_flag))) + int(ab_flag))
    Nb = int(np.round((prec+1)*(eb-wb+int(ab_flag)))    + int(not ab_flag))
    return (np.linspace(a_bds[0], a_bds[-1], Na, endpoint=ab_flag),
            np.linspace(b_bds[0], b_bds[-1], Nb, endpoint=not ab_flag))

def find_max_opt(spl, bounds, prec_r, prec_phi):
    neg_spl = lambda x: float(-spl(x[0], x[1]))
    neg_jac = lambda x: np.array([-float(spl(x[0], x[1], dtheta=1)),
                                   -float(spl(x[0], x[1], dphi=1)) / np.sin(x[0])])
    lats_g = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], 0, bounds['lat_pole_flag'])
    if bounds['lon_idxs'] is None:
        la, lb = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'],
                             bounds['lon_a_idxs'], bounds['lon_b_idxs'], 0, bounds['lon_ab_flag'])
        lons_g = np.concatenate([_unwrap(la), _unwrap(lb)])
        phi_bds = (_unwrap(lb)[0], _unwrap(la)[-1]) if bounds['lon_ab_flag'] else (_unwrap(la)[0], _unwrap(lb)[-1])
    else:
        lons_g  = _unwrap(lon_test_std(bounds['lon_bds'], bounds['lon_idxs'], 0, bounds['lon_std_flag']))
        phi_bds = (lons_g[0], lons_g[-1])
    lat_bds = (lats_g[0], lats_g[-1])

    best_val, best_lat, best_lon = -np.inf, lats_g[0], lons_g[0]
    for lat in lats_g:
        for lon in lons_g:
            res = minimize(neg_spl, [lat, lon], jac=neg_jac, method='L-BFGS-B',
                           bounds=[lat_bds, phi_bds], tol=1e-3)
            val = -res.fun
            if val > best_val:
                best_val, best_lat, best_lon = val, res.x[0], res.x[1]
    if bounds['lat_pole_flag'] and float(spl(0,0)) > best_val:
        best_lat  = 0.0
        best_lon  = rand.uniform(0, 2*np.pi)
        best_val  = float(spl(0,0))
    if best_lon < 0:
        best_lon += 2*np.pi
    return best_val, best_lat, th_to_r(best_lat), best_lon

def find_max_sample(spl, bounds, prec_r, prec_phi):
    lats_t = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], prec_r, bounds['lat_pole_flag'])
    if bounds['lon_idxs'] is None:
        la, lb = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'],
                             bounds['lon_a_idxs'], bounds['lon_b_idxs'], prec_phi, bounds['lon_ab_flag'])
        la, lb = _unwrap(la), _unwrap(lb)
        da, db = spl(lats_t, la), spl(lats_t, lb)
        if bounds['lon_ab_flag']:
            data_t = np.hstack((db, da)); lons_t = np.concatenate([lb, la])
        else:
            data_t = np.hstack((da, db)); lons_t = np.concatenate([la, lb])
    else:
        lons_t = _unwrap(lon_test_std(bounds['lon_bds'], bounds['lon_idxs'], prec_phi, bounds['lon_std_flag']))
        data_t = spl(lats_t, lons_t)
    best_val = np.max(data_t)
    li, lni  = np.unravel_index(np.argmax(data_t), data_t.shape)
    best_lat, best_lon = lats_t[li], lons_t[lni]
    if bounds['lat_pole_flag'] and float(spl(0,0)) > best_val:
        best_lat = 0.0; best_lon = rand.uniform(0, 2*np.pi); best_val = float(spl(0,0))
    if best_lon < 0:
        best_lon += 2*np.pi
    return best_val, best_lat, th_to_r(best_lat), best_lon

def bins_r(r_g, prec, width, r_idx_outer):
    """Build radial bins. n_test_per_bin is informational only for the
    current single-point-per-frame histogramming approach."""
    n_g     = r_idx_outer + 1
    r_g_aug = np.concatenate(([0], r_g[:n_g]))
    test_pts = np.concatenate([
        np.linspace(r_g_aug[i], r_g_aug[i+1], prec+1, endpoint=False)
        for i in range(len(r_g_aug)-1)] + [[r_g_aug[-1]]])
    n_edges   = int(np.ceil((n_g+1) / width)) + 1
    bin_edges = [0.0]
    for i in range(1, n_edges-1):
        mid = int(width*(i-1)*(prec+1)) + prec//2
        bin_edges.append(0.5*(test_pts[mid] + test_pts[mid+1]))
    bin_edges.append(r_g_aug[-1])
    bin_edges   = np.array(bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    drs         = np.diff(bin_edges)
    n_test = np.array([np.sum((test_pts > bin_edges[i]) & (test_pts < bin_edges[i+1]))
                       for i in range(len(bin_edges)-1)])
    if np.unique(n_test).size > 1:
        print(f"Note: non-uniform r bin test counts {np.unique(n_test)} "
              f"(informational only for current histogramming approach).")
    te = np.concatenate([[0], [0.5*(test_pts[k-1]+test_pts[k]) for k in range(1,len(test_pts))],
                         [test_pts[-1]+(test_pts[-1]-test_pts[-2])]])
    return bin_centers, bin_edges, drs, n_test, test_pts, te

def bins_phi(phi_g, prec, width):
    n_g      = phi_g.shape[0]
    test_pts = np.linspace(0, 2*np.pi, n_g*(prec+1), endpoint=False)
    n_edges  = int(np.ceil(n_g / width))
    me = []
    for i in range(1, n_edges):
        me.append(0.5*(phi_g[int(width*(i-1))] + phi_g[int(width*(i-1))+1]))
    last = int(width*(n_edges-1))
    me.append(0.5*(phi_g[last] + phi_g[last+1]) if last+1 <= n_g-1 else 0.5*(phi_g[last]+2*np.pi))
    me   = np.array(me)
    e0a  = np.array([me[-1], 2*np.pi])
    e0b  = np.array([0, me[0]])
    n_t  = np.array([np.sum((test_pts > me[i]) & (test_pts < me[i+1])) for i in range(n_edges-1)]
                    + [np.sum((test_pts > e0a[0]) | (test_pts < e0b[1]))])
    if np.unique(n_t).size > 1:
        print(f"Note: non-uniform phi bin test counts {np.unique(n_t)}.")
    cen  = np.array([0.5*(me[i]+me[i+1]) for i in range(n_edges-1)]
                    + [np.mod(0.5*(me[-1]+2*np.pi+me[0]), 2*np.pi)])
    dph  = np.concatenate((np.diff(me), [(2*np.pi+me[0])-me[-1]]))
    tem  = np.linspace(0.5*(test_pts[0]+test_pts[1]), 0.5*(test_pts[-1]+2*np.pi), n_g*(prec+1))
    return cen, me, e0a, e0b, dph, n_t, test_pts, tem, np.array([tem[-1],2*np.pi]), np.array([0.,tem[0]])

### open file and select writes ###
f     = h5py.File(file_str, 'r')
t_all = f['tasks/KE'].dims[0]['sim_time'][:]

# clip requested window to available data — avoids IndexError on out-of-range times
t_out_end  = min(t_out_end,  t_all[-1])
t_hist_end = min(t_hist_end, t_all[-1])
if t_out_start > t_all[-1]:
    raise ValueError(f"t_out_start={t_out_start} beyond last available time {t_all[-1]:.3f}")

ws_start = np.where(t_all <= t_out_start)[0][-1] if np.any(t_all <= t_out_start) else 0
ws_end   = np.where(t_all >= t_out_end)[0][0]
ws       = np.arange(ws_start, ws_end + 1)
nw, tw   = len(ws), t_all[ws]
print(f"Processing {nw} writes: t={tw[0]:.3f} to t={tw[-1]:.3f}")

### determine r_cutoff ###
if use_cutoff and not r_cutoff_given:
    tdur     = min((t_all[-1]-t_all[0])/3, 1/alpha)
    si       = np.where(t_all >= t_all[-1]-tdur)[0][0]
    EN_tavg  = np.mean(f['tasks/EN'][si:-1])
    KE_tavg  = ((eps/np.pi) - nu*EN_tavg) / (2*alpha)
    u_rms    = np.sqrt(2*KE_tavg)
    r_cutoff = 2*(u_rms/gamma)**(1/3) if gamma > 0 else 0.9
print(f"r_cutoff = {r_cutoff:.4f}" if use_cutoff else "r_cutoff: disabled")

if use_cutoff:
    cutoff_mask  = r_mesh >= r_cutoff
    r_cutoff_idx = np.where(r_1d < r_cutoff)[0][-1]

### check for velocity tasks ###
has_vel = 'u' in f['tasks'] or ('ur' in f['tasks'] and 'uphi' in f['tasks'])
if extract_vel and not has_vel:
    print("Warning: velocity fields not found in HDF5 tasks. Disabling velocity extraction.")
    extract_vel = False

r_profile_max = r_cutoff if use_cutoff else 1.0
r_profile_pts = np.linspace(0, r_profile_max, n_r_profile)
if extract_vel:
    uphi_CPC_sum  = np.zeros(n_r_profile)
    ur_CPC_sum    = np.zeros(n_r_profile)
    n_vel_profile = 0

### RW subtraction setup ###
if subtract_rw:
    # load EVP eigenvector
    evp_data     = np.load(evp_file, allow_pickle=True)[()]
    evals        = evp_data['evals_res']
    evecs_psi    = evp_data['psi_right_evecs_res']
    sort_idxs    = np.argsort(evals.imag)
    psi_evec_real = evecs_psi[sort_idxs][evp_mode_idx].real
    print(f"EVP mode {evp_mode_idx}: eval={evals[sort_idxs][evp_mode_idx].real:.4f}"
          f"+i{evals[sort_idxs][evp_mode_idx].imag:.4f}")

    # load projection time series — projdot_c and projdot_s for dominant mode
    proj_data  = np.load(proj_file, allow_pickle=True)[()]
    proj_tw    = proj_data['tw']           # time axis of projection output
    proj_c     = proj_data['projdot_c'][0] # cosine projection, shape (nw_proj,)
    proj_s     = proj_data['projdot_s'][0] # sine   projection, shape (nw_proj,)
    print(f"Projection file loaded: {len(proj_tw)} frames, "
          f"t=[{proj_tw[0]:.3f}, {proj_tw[-1]:.3f}]")

    # build Dedalus fields for RW streamfunction and its velocity derivatives
    psi_rw_field = dist.Field(bases=disk)

    # unit vector fields for extracting gradient components
    er_rw   = dist.VectorField(coords, bases=disk)
    ephi_rw = dist.VectorField(coords, bases=disk)
    er_rw['g'][1]   = 1.0   # radial unit vector
    ephi_rw['g'][0] = 1.0   # azimuthal unit vector

    # accumulate RW-subtracted profiles separately
    uphi_CPC_sub_sum = np.zeros(n_r_profile)
    ur_CPC_sub_sum   = np.zeros(n_r_profile)
    n_vel_sub        = 0

rand = np.random.RandomState(seed=10101)
Nphi_deal = int(np.round(dealias*Nphi))
Nr_deal   = int(np.round(dealias*Nr))

### tracking loop ###
vort_mus     = []
vort_stddevs = []
lat_poi_idxs = []
lon_poi_idxs = []
lat_pois     = []
lon_pois     = []
vort_maxs    = []
th_locs      = []
r_locs       = []
phi_locs     = []
glitch_flags = []

r_prev    = None
phi_prev  = None
vort_prev = None

prog_cad = max(1, nw // 50)
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"writes loop: i={i} out of {nw}")

    vort.load_from_hdf5(f, w)
    vort_g = np.copy(vort['g'])

    mu_fit, stddev_fit = norm.fit(vort_g)
    vort_mus.append(mu_fit)
    vort_stddevs.append(stddev_fit)

    if use_cutoff:
        vort_g[cutoff_mask] = 0.
    if use_stddev:
        vort_g[np.abs(vort_g) <= 2*stddev_fit] = 0.

    lon_poi_idx = np.where(vort_g == np.max(vort_g))[0][0]
    lat_poi_idx = np.where(vort_g == np.max(vort_g))[1][0]

    # refine cutoff if maximum sits at boundary
    if use_cutoff and lat_poi_idx == r_cutoff_idx:
        vort_g_ref   = np.copy(vort['g'])
        r_cut_ref    = r_cutoff
        success      = False
        while r_cut_ref >= 0.1:
            r_cut_ref *= 0.9
            ri_ref     = np.where(r_1d < r_cut_ref)[0][-1]
            print(f"  frame {i}: refining cutoff to {r_cut_ref:.4f}")
            vort_g_ref[r_mesh >= r_cut_ref] = 0.
            nli = np.where(vort_g_ref == np.max(vort_g_ref))[0][0]
            nri = np.where(vort_g_ref == np.max(vort_g_ref))[1][0]
            if nri != ri_ref:
                lon_poi_idx, lat_poi_idx = nli, nri
                vort_g = vort_g_ref
                print("  refine successful")
                success = True
                break
        if not success:
            print("  refine unsuccessful")

    lat_poi_idxs.append(lat_poi_idx)
    lon_poi_idxs.append(lon_poi_idx)
    lat_pois.append(theta_mesh[lon_poi_idx, lat_poi_idx])
    lon_pois.append(phi_mesh[lon_poi_idx, lat_poi_idx])

    # spline refinement
    if use_interp:
        lm, la, ds, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g,
                                         lon_poi_idx, lat_poi_idx,
                                         Nphi_deal, Nr_deal, local_size_phi, local_size_r)
        lats_spl = la[0, :]
        lons_spl = lm[:, 0].copy()
        lons_spl[lons_spl >= np.pi] -= 2*np.pi
        resort   = np.argsort(lons_spl)
        spl_out  = splinefit(lats_spl, lons_spl[resort], ds[resort,:].T, pole_continuity=True)
        if use_optimize:
            data_max, lat_loc, r_loc, lon_loc = find_max_opt(spl_out, bounds, precision_r, precision_phi)
        else:
            data_max, lat_loc, r_loc, lon_loc = find_max_sample(spl_out, bounds, precision_r, precision_phi)
    else:
        data_max = np.max(vort_g)
        lat_loc  = theta_mesh[lon_poi_idx, lat_poi_idx]
        lon_loc  = phi_mesh[lon_poi_idx, lat_poi_idx]
        r_loc    = th_to_r(lat_loc)

    # temporal continuity glitch check
    glitch = False
    if r_prev is not None:
        dr   = abs(r_loc - r_prev)
        dphi = abs(np.arctan2(np.sin(lon_loc - phi_prev), np.cos(lon_loc - phi_prev)))
        if dr > max_jump_r or dphi > max_jump_phi:
            if vort_prev is not None and data_max < jump_vort_fac * vort_prev:
                glitch = True
                print(f"  frame {i}: glitch — dr={dr:.4f}, dphi={dphi:.4f}, "
                      f"vort={data_max:.2f} vs prev={vort_prev:.2f}")

    glitch_flags.append(glitch)
    vort_maxs.append(data_max)
    th_locs.append(lat_loc)
    r_locs.append(r_loc)
    phi_locs.append(lon_loc)
    r_prev    = r_loc
    phi_prev  = lon_loc
    vort_prev = data_max

    ### velocity profile ###
    if extract_vel and not glitch:
        try:
            u_field.load_from_hdf5(f, w)
            u_field.change_scales(dealias)
            ur_g   = u_field['g'][1]   # radial component,    shape (Nphi_deal, Nr_deal)
            uphi_g = u_field['g'][0]   # azimuthal component, shape (Nphi_deal, Nr_deal)
        except Exception as e:
            print(f"  frame {i}: velocity load failed ({e}), skipping")
            continue

        # exact CPC-frame transformation (from 5_7_26 notes)
        # u_phi^CPC = u_r^lab * sin(phi - phi_CPC) + u_phi^lab * cos(phi - phi_CPC)
        # u_r^CPC   = u_r^lab * cos(phi - phi_CPC) - u_phi^lab * sin(phi - phi_CPC)
        dphi_field     = phi_mesh - lon_loc
        uphi_CPC_field = ur_g * np.sin(dphi_field) + uphi_g * np.cos(dphi_field)
        ur_CPC_field   = ur_g * np.cos(dphi_field) - uphi_g * np.sin(dphi_field)

        # extract along radial slice closest to phi_CPC
        phi_slice_idx = np.argmin(np.abs(phi_1d - lon_loc))
        uphi_CPC_sum += np.interp(r_profile_pts, r_1d, uphi_CPC_field[phi_slice_idx, :])
        ur_CPC_sum   += np.interp(r_profile_pts, r_1d, ur_CPC_field[phi_slice_idx, :])
        n_vel_profile += 1

        ### RW-subtracted velocity profile ###
        if subtract_rw:
            # find projection amplitude and phase at this frame by interpolating onto tw[i]
            t_now   = tw[i]
            proj_c_i = float(np.interp(t_now, proj_tw, proj_c))
            proj_s_i = float(np.interp(t_now, proj_tw, proj_s))
            A_inst   = np.sqrt(proj_c_i**2 + proj_s_i**2)
            # instantaneous wave phase: phi_wave = atan2(-proj_s/norm, proj_c/norm)
            # since both are scaled by the same norm, the ratio is correct as-is
            phi_wave_inst = np.arctan2(-proj_s_i, proj_c_i)

            # build instantaneous RW streamfunction:
            # psi_RW(r, phi, t) = A(t) * Re[psi_evec(r) * exp(i*m*(phi - phi_wave(t)))]
            # for m=1: psi_RW = A(t) * psi_evec_real(r, phi - phi_wave(t))
            # we implement the phase shift by rotating the phi argument of the stored evec
            # psi_evec_real is on the dealiased grid with shape (Nphi_deal, Nr_deal)
            # the evec was computed at phi_wave=0 (cos basis), so we shift phi by phi_wave_inst
            phi_shifted = phi_mesh - phi_wave_inst   # shift so evec aligns with wave
            # interpolate evec onto shifted phi grid using the stored grid values
            # since evec is periodic, we use np.interp with wrapping
            phi_evec_grid = np.linspace(0, 2*np.pi, psi_evec_real.shape[0], endpoint=False)
            psi_rw_g = np.zeros_like(phi_mesh)
            for ri in range(phi_mesh.shape[1]):
                evec_col = psi_evec_real[:, ri]
                phi_s_col = np.mod(phi_shifted[:, ri], 2*np.pi)
                psi_rw_g[:, ri] = A_inst * np.interp(phi_s_col, phi_evec_grid, evec_col)

            # RW velocity from psi_RW:
            # u = -d3.skew(d3.grad(psi)) gives (u_phi, u_r) = (dr(psi), -1/r*dphi(psi))
            # so u_phi_RW is component [0] and u_r_RW is component [1]
            psi_rw_field.change_scales(dealias)
            psi_rw_field['g'] = psi_rw_g
            u_rw = (-d3.skew(d3.grad(psi_rw_field))).evaluate()
            u_rw.change_scales(dealias)
            uphi_rw_g = u_rw['g'][0]   # u_phi_RW = dr(psi_RW)
            ur_rw_g   = u_rw['g'][1]   # u_r_RW   = -1/r * dphi(psi_RW)

            # subtract RW from lab frame velocity
            ur_sub   = ur_g   - ur_rw_g
            uphi_sub = uphi_g - uphi_rw_g

            # apply CPC coordinate transformation to residual
            uphi_CPC_sub = ur_sub * np.sin(dphi_field) + uphi_sub * np.cos(dphi_field)
            ur_CPC_sub   = ur_sub * np.cos(dphi_field) - uphi_sub * np.sin(dphi_field)

            uphi_CPC_sub_sum += np.interp(r_profile_pts, r_1d, uphi_CPC_sub[phi_slice_idx, :])
            ur_CPC_sub_sum   += np.interp(r_profile_pts, r_1d, ur_CPC_sub[phi_slice_idx, :])
            n_vel_sub        += 1

### histograms ###
prec_r_h   = precision_r   if use_interp else 0
prec_phi_h = precision_phi if use_interp else 0

phi_centers, phi_edges_main, phi_edges_0a, phi_edges_0b, dphis, \
    n_t_phi, test_pts_phi, te_main, te_0a, te_0b = bins_phi(phi_1d, prec_phi_h, bin_width_phi)
r_centers, r_edges, drs, n_t_r, test_pts_r, te_r = bins_r(r_1d, prec_r_h, bin_width_r, len(r_1d)-1)

r_inner    = r_edges[:-1];   r_outer    = r_edges[1:]
phi_w_main = phi_edges_main[:-1]; phi_e_main = phi_edges_main[1:]
phi_w_0a, phi_e_0a = phi_edges_0a
phi_w_0b, phi_e_0b = phi_edges_0b

phi_centers_2d = np.tile(phi_centers[:, np.newaxis], (1, len(r_centers)))
r_centers_2d   = np.tile(r_centers[np.newaxis, :],   (len(phi_centers), 1))

hist_r   = np.zeros(len(r_centers))
hist_phi = np.zeros(len(phi_centers))
hist_2d  = np.zeros((len(phi_centers), len(r_centers)))
n_hist   = 0

# hist window — clipped to tracking window
hs = float(t_hist_start)
he = float(t_hist_end)
hs = max(hs, tw[0]); he = min(he, tw[-1])
ws_hist_start = np.where(tw <= hs)[0][-1] if np.any(tw <= hs) else 0
ws_hist_end   = np.where(tw >= he)[0][0]
ws_hist       = np.arange(ws_hist_start, ws_hist_end + 1)

for j in ws_hist:
    if glitch_flags[j]:
        continue
    rl  = r_locs[j]
    phl = phi_locs[j]
    rm   = (rl > r_inner) & (rl < r_outer)
    pm   = np.concatenate(((phl > phi_w_main) & (phl < phi_e_main),
                            [(phl > phi_w_0a and phl < phi_e_0a) or
                             (phl >= phi_w_0b and phl < phi_e_0b)]))
    hist_r[rm]   += 1
    hist_phi[pm] += 1
    hist_2d[np.ix_(pm, rm)] += 1
    n_hist += 1

n_h     = max(n_hist, 1)
pdf_r   = hist_r   / n_h / drs
pdf_phi = hist_phi / n_h / dphis
# correct 2D bin area in polar coords: dA_ij = dphi_i * (r_outer_j^2 - r_inner_j^2) / 2
# ensures integ(pdf_2d * r dr dphi) = 1, consistent with pdf_r and pdf_phi
dAs_2d  = np.outer(dphis, 0.5*(r_outer**2 - r_inner**2))
pdf_2d  = hist_2d / n_h / dAs_2d

### Rice fit on clean samples ###
r_clean     = np.array(r_locs)[~np.array(glitch_flags)]
rice_params = rice.fit(r_clean, floc=0.0) if len(r_clean) > 10 else None

### save ###
processed = {
    'nw': nw, 'ws': ws, 'tw': tw,
    'r_deal': r_deal, 'phi_deal': phi_deal,
    'vort_mus': vort_mus, 'vort_stddevs': vort_stddevs,
    'lat_poi_idxs': lat_poi_idxs, 'lon_poi_idxs': lon_poi_idxs,
    'lat_pois': lat_pois, 'lon_pois': lon_pois,
    'vort_maxs': vort_maxs,
    'th_locs': th_locs, 'r_locs': r_locs, 'phi_locs': phi_locs,
    'glitch_flags': glitch_flags,
    'n_hist': n_hist, 'ws_hist': ws_hist,
    'r_centers': r_centers, 'r_edges': r_edges, 'drs': drs,
    'phi_centers': phi_centers,
    'phi_edges_main': phi_edges_main, 'phi_edges_0a': phi_edges_0a,
    'phi_edges_0b': phi_edges_0b, 'dphis': dphis,
    'r_centers_2d': r_centers_2d, 'phi_centers_2d': phi_centers_2d,
    'hist_r': hist_r, 'hist_phi': hist_phi, 'hist_2d': hist_2d,
    'pdf_r': pdf_r,   'pdf_phi': pdf_phi,   'pdf_2d': pdf_2d,
}
if rice_params is not None:
    processed['rice_fit'] = rice_params
if extract_vel and n_vel_profile > 0:
    processed['uphi_CPC_mean'] = uphi_CPC_sum  / n_vel_profile
    processed['ur_CPC_mean']   = ur_CPC_sum    / n_vel_profile
    processed['r_profile_pts'] = r_profile_pts
    processed['n_vel_profile'] = n_vel_profile
    print(f"Velocity profile averaged over {n_vel_profile} non-glitch frames.")
    if subtract_rw and n_vel_sub > 0:
        processed['uphi_CPC_sub_mean'] = uphi_CPC_sub_sum / n_vel_sub
        processed['ur_CPC_sub_mean']   = ur_CPC_sub_sum   / n_vel_sub
        processed['n_vel_sub']         = n_vel_sub
        print(f"RW-subtracted profile averaged over {n_vel_sub} frames.")

out_path = f"{output_prefix}_{output_suffix}.npy"
print(f"Saving to: {out_path}")
np.save(out_path, processed)

"""
Usage:
    process_tracking_v7.py <file1> <file2> <file3> [options]

Arguments:
    <file1>    HDF5 analysis file from Dedalus simulation
    <file2>    processed projection FFT .npy file
    <file3>    processed EVP .npy file

Options:
    --output=<str>           prefix for output file [default: processed_tracking]
    --t_start=<float>        sim time to begin tracking and profile accumulation [default: 149.]
    --t_end=<float>          sim time to stop tracking and profile accumulation [default: 251.]

    --use_cutoff=<bool>      ignore grid data beyond a specified radius [default: True]
    --r_cutoff=<float>       cutoff radius; if None, computed from Lgamma [default: None]
    --use_stddev=<bool>      ignore grid data below a multiple of the standard deviation [default: False]
    --bin_width_phi=<int>    dedalus grid points per bin in phi [default: 1]
    --bin_width_r=<int>      dedalus grid points per bin in r [default: 1]

    --use_interp=<bool>      use bivariate spline to refine extremum location [default: True]
    --use_optimize=<bool>    use optimization rather than sampling to find spline extremum [default: True]
    --local_size_phi=<int>   phi grid points passed to spline (>= 2) [default: 2]
    --local_size_r=<int>     r grid points passed to spline (>= 2, recommend >= 3) [default: 3]
    --precision_phi=<int>    even number of sample points per phi interval for spline [default: 2]
    --precision_r=<int>      even number of sample points per r interval for spline [default: 4]

    --max_jump_r=<float>     max allowed jump in r between frames before flagging as glitch [default: 0.1]
    --max_jump_phi=<float>   max allowed jump in phi (rad) between frames before flagging as glitch [default: 1.5]
    --jump_vort_fac=<float>  if jump detected, only accept if new vorticity exceeds old by this factor [default: 2.0]

    --rho_max=<float>        outer radius of CPC-frame polar grid [default: 0.3]
    --rho_min=<float>        inner radius of CPC-frame polar grid (log spacing) [default: 1e-3]
    --n_rho=<int>            number of radial points (log-spaced) in CPC-frame grid [default: 256]
    --n_phi_cpc=<int>        number of azimuthal points in CPC-frame polar grid [default: 256]

    --subtract_rw=<bool>     subtract dominant RW from velocity before CPC transform [default: True]
    --evp_mode_idx=<int>     index of dominant EVP mode (sort_im_inc order) [default: 1]
    --fft_mode_idx=<int>     amplitude-sorted FFT mode index for orbital velocity correction [default: 0]
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
from scipy.interpolate import RectSphereBivariateSpline as splinefit
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm, rice
from scipy.optimize import minimize

print("args read in")
print(args)

file_str      = args['<file1>']
proj_file     = args['<file2>']
evp_file      = args['<file3>']
output_prefix = args['--output']

t_start = float(args['--t_start'])
t_end   = float(args['--t_end'])

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

rho_max   = float(args['--rho_max'])
rho_min   = float(args['--rho_min'])
n_rho     = int(args['--n_rho'])
n_phi_cpc = int(args['--n_phi_cpc'])

subtract_rw  = eval(args['--subtract_rw'])
evp_mode_idx = int(args['--evp_mode_idx'])
fft_mode_idx = int(args['--fft_mode_idx'])

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
Nphi       = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr         = int(output_suffix.split('Nr_')[1].split('_')[0])
alpha_read = str_to_float(output_suffix.split('alpha_')[1].split('_')[0])
gamma_read = str_to_float(output_suffix.split('gam_')[1].split('_')[0])
eps_read   = str_to_float(output_suffix.split('eps_')[1].split('_')[0])
nu_read    = str_to_float(output_suffix.split('nu_')[1].split('_')[0])

alpha_vals = np.array((2e-3, 1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
eps_vals   = np.array([3.3e-1, 1.0, 2.0])
nu_vals    = np.array([5e-5, 8/90000, 2e-4])
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
vort    = dist.Field(name='vort', bases=disk)
u_field = dist.VectorField(coords, name='u', bases=disk)

phi_1d = phi_deal[:, 0]   # (Nphi_deal,)
r_1d   = r_deal[0, :]     # (Nr_deal,)
Nphi_deal = len(phi_1d)
Nr_deal   = len(r_1d)

phi_mesh   = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))
theta_1d   = r_to_th(r_1d)
theta_mesh = np.tile(theta_1d[np.newaxis, :], (Nphi_deal, 1))
r_mesh     = np.tile(r_1d[np.newaxis, :],     (Nphi_deal, 1))

# wrapped phi axis for periodic interpolation
phi_1d_wrap = np.append(phi_1d, 2*np.pi)

### spline helper functions ###
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
    neg_spl = lambda x: float(-spl(x[0], x[1]).ravel()[0])
    neg_jac = lambda x: np.array([-float(spl(x[0], x[1], dtheta=1).ravel()[0]),
                                   -float(spl(x[0], x[1], dphi=1).ravel()[0]) / np.sin(x[0])])
    lats_g = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], 0, bounds['lat_pole_flag'])
    if bounds['lon_idxs'] is None:
        la, lb = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'],
                             bounds['lon_a_idxs'], bounds['lon_b_idxs'], 0, bounds['lon_ab_flag'])
        lons_g  = np.concatenate([_unwrap(la), _unwrap(lb)])
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
    pole_val = float(spl(0, 0).ravel()[0])
    if bounds['lat_pole_flag'] and pole_val > best_val:
        best_lat = 0.0
        best_lon = rand.uniform(0, 2*np.pi)
        best_val = pole_val
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
    pole_val = float(spl(0, 0).ravel()[0])
    if bounds['lat_pole_flag'] and pole_val > best_val:
        best_lat = 0.0; best_lon = rand.uniform(0, 2*np.pi); best_val = pole_val
    if best_lon < 0:
        best_lon += 2*np.pi
    return best_val, best_lat, th_to_r(best_lat), best_lon

### histogram bin helpers ###
def bins_r(r_g, prec, width, r_idx_outer):
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
        print(f"Note: non-uniform r bin test counts {np.unique(n_test)}.")
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

t_end = min(t_end, t_all[-1])
if t_start > t_all[-1]:
    raise ValueError(f"t_start={t_start} beyond last available time {t_all[-1]:.3f}")

ws_start = np.where(t_all <= t_start)[0][-1] if np.any(t_all <= t_start) else 0
ws_end   = np.where(t_all >= t_end)[0][0]
ws       = np.arange(ws_start, ws_end + 1)
nw, tw   = len(ws), t_all[ws]
print(f"Processing {nw} writes: t={tw[0]:.3f} to t={tw[-1]:.3f}")

### determine r_cutoff ###
if use_cutoff and not r_cutoff_given:
    tdur    = min((t_all[-1]-t_all[0])/3, 1/alpha)
    si      = np.where(t_all >= t_all[-1]-tdur)[0][0]
    EN_tavg = np.mean(f['tasks/EN'][si:-1])
    KE_tavg = ((eps/np.pi) - nu*EN_tavg) / (2*alpha)
    u_rms   = np.sqrt(2*KE_tavg)
    if not np.isfinite(u_rms):
        KE_tavg = np.mean(f['tasks/KE'][si:-1])
        print("using KE analysis task to estimate r_cutoff")
        u_rms   = np.sqrt(2*KE_tavg)
    r_cutoff = 2*(u_rms/gamma)**(1/3) if gamma > 0 else 0.9
print(f"r_cutoff = {r_cutoff:.4f}" if use_cutoff else "r_cutoff: disabled")

if use_cutoff:
    cutoff_mask  = r_mesh >= r_cutoff
    r_cutoff_idx = np.where(r_1d < r_cutoff)[0][-1]

### check velocity tasks ###
has_vel = 'u' in f['tasks'] or ('ur' in f['tasks'] and 'uphi' in f['tasks'])
if not has_vel:
    raise RuntimeError("No velocity fields found in HDF5 file.")

### load projection file and extract omega for orbital correction ###
proj_data = np.load(proj_file, allow_pickle=True)[()]
proj_tw   = proj_data['tw']
proj_c    = proj_data['projdot_c'][0]
proj_s    = proj_data['projdot_s'][0]
print(f"Projection file loaded: {len(proj_tw)} frames, "
      f"t=[{proj_tw[0]:.3f}, {proj_tw[-1]:.3f}]")

fft_sel_idx = proj_data['fft_selected_idx'][0]
omega       = proj_data['fft_peak_freqs'][fft_mode_idx, fft_sel_idx]
print(f"Orbital correction: omega={omega:.6f} (fft_mode_idx={fft_mode_idx})")

### RW subtraction setup ###
if subtract_rw:
    evp_data      = np.load(evp_file, allow_pickle=True)[()]
    evals         = evp_data['evals_res']
    evecs_psi     = evp_data['psi_right_evecs_res']
    sort_idxs     = np.argsort(evals.imag)
    psi_evec_real = evecs_psi[sort_idxs][evp_mode_idx].real
    print(f"EVP mode {evp_mode_idx}: eval={evals[sort_idxs][evp_mode_idx].real:.4f}"
          f"+i{evals[sort_idxs][evp_mode_idx].imag:.4f}")

    psi_rw_field = dist.Field(bases=disk)

### CPC-frame polar grid ###
rho_pts   = np.geomspace(rho_min, rho_max, n_rho)
alpha_pts = np.linspace(0, 2*np.pi, n_phi_cpc, endpoint=False)
rho_mesh_cpc, alpha_mesh_cpc = np.meshgrid(rho_pts, alpha_pts, indexing='ij')
# (n_rho, n_phi_cpc)

### profile accumulators ###
uphi_2d_raw_sum  = np.zeros(n_rho)
uphi_2d_corr_sum = np.zeros(n_rho)
ur_2d_sum        = np.zeros(n_rho)
n_az_samples     = np.zeros(n_rho, dtype=int)

if subtract_rw:
    uphi_2d_sub_raw_sum  = np.zeros(n_rho)
    uphi_2d_sub_corr_sum = np.zeros(n_rho)
    ur_2d_sub_sum        = np.zeros(n_rho)
    n_az_sub_samples     = np.zeros(n_rho, dtype=int)

### tracking state ###
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

n_frames_used    = 0
n_frames_skipped = 0

rand     = np.random.RandomState(seed=10101)
prog_cad = max(1, nw // 50)

### main loop ###
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"writes loop: i={i} out of {nw}")

    # ── vorticity tracking ────────────────────────────────────────────────────
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

    if use_cutoff and lat_poi_idx == r_cutoff_idx:
        vort_g_ref = np.copy(vort['g'])
        r_cut_ref  = r_cutoff
        success    = False
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

    if glitch:
        n_frames_skipped += 1
        continue

    # ── load velocity ─────────────────────────────────────────────────────────
    try:
        u_field.load_from_hdf5(f, w)
        u_field.change_scales(dealias)
        ur_g   = u_field['g'][1]   # (Nphi_deal, Nr_deal)
        uphi_g = u_field['g'][0]
    except Exception as e:
        print(f"  frame {i}: velocity load failed ({e}), skipping")
        n_frames_skipped += 1
        continue

    # ── orbital velocity correction (lab frame) ───────────────────────────────
    # The wave phase drifts as phi(t) = phi_0 - omega*t (for m=1), so the CPC
    # is carried in the -phi direction at speed omega*r_loc in the lab frame.
    # v_orb is defined as the negative of that drift speed; subtracting v_orb
    # from u_phi^lab removes the orbital contribution before the CPC transform.
    v_orb        = -omega * r_loc   # kept for metadata/reference only
    uphi_g_raw   = uphi_g          # original lab-frame uphi, no correction
    uphi_g_corr  = uphi_g          # correction applied pointwise after interpolation

    # ── build RW velocity and subtract ───────────────────────────────────────
    if subtract_rw:
        t_now         = tw[i]
        proj_c_i      = float(np.interp(t_now, proj_tw, proj_c))
        proj_s_i      = float(np.interp(t_now, proj_tw, proj_s))
        A_inst        = np.sqrt(proj_c_i**2 + proj_s_i**2)
        phi_wave_inst = np.arctan2(-proj_s_i, proj_c_i)

        phi_evec_grid = np.linspace(0, 2*np.pi, psi_evec_real.shape[0], endpoint=False)
        psi_rw_g      = np.zeros((Nphi_deal, Nr_deal))
        for ri in range(Nr_deal):
            phi_s_col = np.mod(phi_mesh[:, ri] - phi_wave_inst, 2*np.pi)
            psi_rw_g[:, ri] = A_inst * np.interp(phi_s_col, phi_evec_grid,
                                                   psi_evec_real[:, ri])

        psi_rw_field.change_scales(dealias)
        psi_rw_field['g'] = psi_rw_g
        u_rw = (-d3.skew(d3.grad(psi_rw_field))).evaluate()
        u_rw.change_scales(dealias)
        uphi_rw_g = u_rw['g'][0]
        ur_rw_g   = u_rw['g'][1]

        ur_sub_g      = ur_g        - ur_rw_g
        uphi_sub_g_raw  = uphi_g_raw  - uphi_rw_g
        uphi_sub_g_corr = uphi_g_corr - uphi_rw_g

    # ── map CPC polar grid to lab polar coordinates ───────────────────────────
    x_cpc = r_loc * np.cos(lon_loc)
    y_cpc = r_loc * np.sin(lon_loc)

    x_lab = x_cpc + rho_mesh_cpc * np.cos(alpha_mesh_cpc)   # (n_rho, n_phi_cpc)
    y_lab = y_cpc + rho_mesh_cpc * np.sin(alpha_mesh_cpc)

    r_lab   = np.sqrt(x_lab**2 + y_lab**2)
    phi_lab = np.arctan2(y_lab, x_lab) % (2*np.pi)

    # validity: inside the dedalus grid domain
    r_bound = r_1d[-1]
    valid   = r_lab <= r_bound   # (n_rho, n_phi_cpc)

    # ── orbital velocity correction (lab frame) ───────────────────────────────
    # The wave phase drifts as phi(t) = phi_0 - omega*t (for m=1), so the
    # background solid-body-like rotation contributes omega*r to u_phi in the
    # lab frame. We define v_orb(r) = -omega*r and subtract it pointwise from
    # u_phi^lab at each CPC-frame grid point after interpolation, before the
    # CPC transformation. r_lab has shape (n_rho, n_phi_cpc).

    # ── build interpolators (periodic wrap on phi axis) ───────────────────────
    def make_interp(field_2d):
        field_wrap = np.vstack([field_2d, field_2d[0:1, :]])
        return RegularGridInterpolator(
            (phi_1d_wrap, r_1d), field_wrap,
            method='linear', bounds_error=False, fill_value=np.nan)

    interp_ur        = make_interp(ur_g)
    interp_uphi_raw  = make_interp(uphi_g_raw)
    interp_uphi_corr = make_interp(uphi_g_corr)
    if subtract_rw:
        interp_ur_sub        = make_interp(ur_sub_g)
        interp_uphi_sub_raw  = make_interp(uphi_sub_g_raw)
        interp_uphi_sub_corr = make_interp(uphi_sub_g_corr)

    pts_flat = np.column_stack([phi_lab.ravel(), r_lab.ravel()])

    ur_lab_flat        = interp_ur(pts_flat).reshape(n_rho, n_phi_cpc)
    uphi_raw_flat      = interp_uphi_raw(pts_flat).reshape(n_rho, n_phi_cpc)
    uphi_corr_flat     = interp_uphi_corr(pts_flat).reshape(n_rho, n_phi_cpc)
    if subtract_rw:
        ur_sub_lab_flat      = interp_ur_sub(pts_flat).reshape(n_rho, n_phi_cpc)
        uphi_sub_raw_flat    = interp_uphi_sub_raw(pts_flat).reshape(n_rho, n_phi_cpc)
        uphi_sub_corr_flat   = interp_uphi_sub_corr(pts_flat).reshape(n_rho, n_phi_cpc)

    # apply solid-body orbital correction pointwise in r to the corr fields
    uphi_corr_flat -= omega * r_lab
    if subtract_rw:
        uphi_sub_corr_flat -= omega * r_lab

    # ── CPC velocity transformation (per grid point) ──────────────────────────
    # Convert lab polar -> lab Cartesian, then project onto CPC-frame basis.
    # CPC-frame radial direction at grid point (rho, alpha): (cos alpha, sin alpha)
    # CPC-frame azimuthal direction:                        (-sin alpha, cos alpha)
    def polar_to_cpc(ur_flat, uphi_flat):
        u_x = ur_flat * np.cos(phi_lab) - uphi_flat * np.sin(phi_lab)
        u_y = ur_flat * np.sin(phi_lab) + uphi_flat * np.cos(phi_lab)
        uphi_cpc = -u_x * np.sin(alpha_mesh_cpc) + u_y * np.cos(alpha_mesh_cpc)
        ur_cpc   =  u_x * np.cos(alpha_mesh_cpc) + u_y * np.sin(alpha_mesh_cpc)
        return uphi_cpc, ur_cpc

    uphi_cpc_raw_grid,  ur_cpc_grid     = polar_to_cpc(ur_lab_flat, uphi_raw_flat)
    uphi_cpc_corr_grid, _               = polar_to_cpc(ur_lab_flat, uphi_corr_flat)
    if subtract_rw:
        uphi_cpc_sub_raw_grid,  ur_cpc_sub_grid = polar_to_cpc(ur_sub_lab_flat, uphi_sub_raw_flat)
        uphi_cpc_sub_corr_grid, _               = polar_to_cpc(ur_sub_lab_flat, uphi_sub_corr_flat)

    # ── azimuthal average over valid, finite points at each rho ──────────────
    for k in range(n_rho):
        vmask       = valid[k, :]
        uphi_raw_v  = uphi_cpc_raw_grid[k, vmask]
        uphi_corr_v = uphi_cpc_corr_grid[k, vmask]
        ur_v        = ur_cpc_grid[k, vmask]
        good        = np.isfinite(uphi_raw_v) & np.isfinite(ur_v)
        n_good      = np.sum(good)
        if n_good > 0:
            uphi_2d_raw_sum[k]  += np.sum(uphi_raw_v[good])
            uphi_2d_corr_sum[k] += np.sum(uphi_corr_v[good])
            ur_2d_sum[k]        += np.sum(ur_v[good])
            n_az_samples[k]     += n_good

        if subtract_rw:
            uphi_sv_raw  = uphi_cpc_sub_raw_grid[k, vmask]
            uphi_sv_corr = uphi_cpc_sub_corr_grid[k, vmask]
            ur_sv        = ur_cpc_sub_grid[k, vmask]
            good_s       = np.isfinite(uphi_sv_raw) & np.isfinite(ur_sv)
            n_good_s     = np.sum(good_s)
            if n_good_s > 0:
                uphi_2d_sub_raw_sum[k]  += np.sum(uphi_sv_raw[good_s])
                uphi_2d_sub_corr_sum[k] += np.sum(uphi_sv_corr[good_s])
                ur_2d_sub_sum[k]        += np.sum(ur_sv[good_s])
                n_az_sub_samples[k]     += n_good_s

    n_frames_used += 1

f.close()
print(f"Frames used: {n_frames_used}, skipped (glitch/load): {n_frames_skipped}")

### histograms ###
prec_r_h   = precision_r   if use_interp else 0
prec_phi_h = precision_phi if use_interp else 0

phi_centers, phi_edges_main, phi_edges_0a, phi_edges_0b, dphis, \
    n_t_phi, test_pts_phi, te_main, te_0a, te_0b = bins_phi(phi_1d, prec_phi_h, bin_width_phi)
r_centers, r_edges, drs, n_t_r, test_pts_r, te_r = bins_r(r_1d, prec_r_h, bin_width_r, len(r_1d)-1)

r_inner = r_edges[:-1]; r_outer = r_edges[1:]
phi_w_main = phi_edges_main[:-1]; phi_e_main = phi_edges_main[1:]
phi_w_0a, phi_e_0a = phi_edges_0a
phi_w_0b, phi_e_0b = phi_edges_0b

phi_centers_2d = np.tile(phi_centers[:, np.newaxis], (1, len(r_centers)))
r_centers_2d   = np.tile(r_centers[np.newaxis, :],   (len(phi_centers), 1))

hist_r   = np.zeros(len(r_centers))
hist_phi = np.zeros(len(phi_centers))
hist_2d  = np.zeros((len(phi_centers), len(r_centers)))
n_hist   = 0

for j in range(nw):
    if glitch_flags[j]:
        continue
    rl  = r_locs[j]
    phl = phi_locs[j]
    rm  = (rl > r_inner) & (rl < r_outer)
    pm  = np.concatenate(((phl > phi_w_main) & (phl < phi_e_main),
                           [(phl > phi_w_0a and phl < phi_e_0a) or
                            (phl >= phi_w_0b and phl < phi_e_0b)]))
    hist_r[rm]   += 1
    hist_phi[pm] += 1
    hist_2d[np.ix_(pm, rm)] += 1
    n_hist += 1

n_h     = max(n_hist, 1)
pdf_r   = hist_r   / n_h / drs
pdf_phi = hist_phi / n_h / dphis
dAs_2d  = np.outer(dphis, 0.5*(r_outer**2 - r_inner**2))
pdf_2d  = hist_2d / n_h / dAs_2d

### Rice fit ###
r_clean     = np.array(r_locs)[~np.array(glitch_flags)]
rice_params = rice.fit(r_clean, floc=0.0) if len(r_clean) > 10 else None

### normalise profiles ###
safe              = n_az_samples > 0
uphi_2d_raw_mean  = np.where(safe, uphi_2d_raw_sum  / np.maximum(n_az_samples, 1), np.nan)
uphi_2d_corr_mean = np.where(safe, uphi_2d_corr_sum / np.maximum(n_az_samples, 1), np.nan)
ur_2d_mean        = np.where(safe, ur_2d_sum         / np.maximum(n_az_samples, 1), np.nan)

if subtract_rw:
    safe_sub              = n_az_sub_samples > 0
    uphi_2d_sub_raw_mean  = np.where(safe_sub, uphi_2d_sub_raw_sum  / np.maximum(n_az_sub_samples, 1), np.nan)
    uphi_2d_sub_corr_mean = np.where(safe_sub, uphi_2d_sub_corr_sum / np.maximum(n_az_sub_samples, 1), np.nan)
    ur_2d_sub_mean        = np.where(safe_sub, ur_2d_sub_sum         / np.maximum(n_az_sub_samples, 1), np.nan)

### save ###
processed = {
    # tracking
    'nw': nw, 'ws': ws, 'tw': tw,
    'r_deal': r_deal, 'phi_deal': phi_deal,
    'vort_mus': vort_mus, 'vort_stddevs': vort_stddevs,
    'lat_poi_idxs': lat_poi_idxs, 'lon_poi_idxs': lon_poi_idxs,
    'lat_pois': lat_pois, 'lon_pois': lon_pois,
    'vort_maxs': vort_maxs,
    'th_locs': th_locs, 'r_locs': r_locs, 'phi_locs': phi_locs,
    'glitch_flags': glitch_flags,
    # histograms
    'n_hist': n_hist,
    'r_centers': r_centers, 'r_edges': r_edges, 'drs': drs,
    'phi_centers': phi_centers,
    'phi_edges_main': phi_edges_main, 'phi_edges_0a': phi_edges_0a,
    'phi_edges_0b': phi_edges_0b, 'dphis': dphis,
    'r_centers_2d': r_centers_2d, 'phi_centers_2d': phi_centers_2d,
    'hist_r': hist_r, 'hist_phi': hist_phi, 'hist_2d': hist_2d,
    'pdf_r': pdf_r, 'pdf_phi': pdf_phi, 'pdf_2d': pdf_2d,
    # 2D azimuthal-average profiles
    'rho_pts': rho_pts, 'alpha_pts': alpha_pts,
    'rho_min': rho_min, 'rho_max': rho_max,
    'n_rho': n_rho, 'n_phi_cpc': n_phi_cpc,
    'uphi_2d_raw_mean':  uphi_2d_raw_mean,
    'uphi_2d_corr_mean': uphi_2d_corr_mean,
    'ur_2d_mean':        ur_2d_mean,
    'n_az_samples': n_az_samples,
    'n_frames_used':    n_frames_used,
    'n_frames_skipped': n_frames_skipped,
    # orbital correction metadata
    'omega': omega,
    'fft_mode_idx': fft_mode_idx,
}
if rice_params is not None:
    processed['rice_fit'] = rice_params
if subtract_rw:
    processed['uphi_2d_sub_raw_mean']  = uphi_2d_sub_raw_mean
    processed['uphi_2d_sub_corr_mean'] = uphi_2d_sub_corr_mean
    processed['ur_2d_sub_mean']        = ur_2d_sub_mean
    processed['n_az_sub_samples']  = n_az_sub_samples

output_str = output_prefix + '_' + output_suffix + '.npy'
output_len = len(output_str)
if output_len > 256:
    print('filename too long, truncating')
    print('before: ' + output_str)
    output_str = (output_str.split('.npy')[0])[:250] + '.npy'
    print('after: ' + output_str)
print("Saving output as:", output_str)
np.save(output_str, processed)

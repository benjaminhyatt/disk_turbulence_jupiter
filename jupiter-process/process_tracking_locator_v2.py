"""
CPC locator and length-scale estimator (Phase A, script 1).

Minimal tracking that identifies the CPC's location (r_CPC(t), phi_CPC(t))
at each frame using the same vorticity-max + optional spline-refinement
approach as v7/v12, plus the same glitch-detection logic. Adds a CPC-frame
azimuthally-averaged radial vorticity profile fit at each frame to estimate a
CPC length scale.

R is reported at two thresholds:
  - R_half:  smallest rho where |omega_bar(rho)| < 0.5 * |omega_bar(0)|  (HWHM-like)
  - R_tenth: smallest rho where |omega_bar(rho)| < 0.1 * |omega_bar(0)|  (effective edge)

Both are robust to whether the vortex is cyclonic or anticyclonic (computed
from absolute values relative to the central peak). The relationship between
the two is itself a diagnostic of profile shape.

Outputs (npy):
  - tracking: tw, r_locs, phi_locs, glitch_flags, vort_maxs, vort_mus,
    vort_stddevs, lat_pois, lon_pois  (keys consistent with v7/v12)
  - R fit:    R_half_history, R_tenth_history, omega_peak_history,
              R_half_mean/std, R_tenth_mean/std, vort_radial_tavg,
              rho_fit_grid, plus the threshold values used.
  - metadata: parsed simulation parameters, time window, options.

Usage:
    process_tracking_locator_v2.py <hdf5_file> [options]

Arguments:
    <hdf5_file>   HDF5 analysis file from Dedalus simulation

Options:
    --output=<str>           output .npy path; 'auto' uses output_suffix [default: auto]
    --plot=<str>             plot path; 'auto' uses output_suffix; 'none' suppresses [default: auto]
    --output_prefix=<str>    prefix used when --output / --plot are 'auto'
                             [default: processed_tracking_locator]

    --t_start=<float>        sim time to begin tracking [default: 149.]
    --t_end=<float>          sim time to stop tracking  [default: 251.]

    --use_cutoff=<bool>      ignore grid data beyond a specified radius [default: True]
    --r_cutoff=<str>         cutoff radius; 'None' computes from Lgamma [default: None]
    --use_stddev=<bool>      ignore grid data below a multiple of stddev [default: False]

    --use_interp=<bool>      use bivariate spline to refine the vortex max [default: True]
    --use_optimize=<bool>    use L-BFGS-B optimization rather than sampling [default: True]
    --local_size_phi=<int>   phi grid points passed to spline (>= 2) [default: 2]
    --local_size_r=<int>     r grid points passed to spline (>= 2, recommend >= 3) [default: 3]
    --precision_phi=<int>    even number of sample points per phi interval [default: 2]
    --precision_r=<int>      even number of sample points per r interval [default: 4]

    --max_jump_r=<float>     max allowed jump in r between frames before glitch [default: 0.1]
    --max_jump_phi=<float>   max allowed jump in phi (rad) between frames [default: 1.5]
    --jump_vort_fac=<float>  vorticity factor for glitch override [default: 2.0]

    --rho_max_fit=<float>    outer rho for the CPC-frame fit grid [default: 0.4]
    --rho_min_fit=<float>    smallest non-zero rho on log-spaced grid [default: 1e-3]
    --n_rho_fit=<int>        number of rho points (log-spaced, with rho=0 prepended) [default: 96]
    --n_alpha_fit=<int>      number of alpha points (uniform spacing) [default: 32]
    --R_threshold_half=<float>   fraction of peak for HWHM-style R [default: 0.5]
    --R_threshold_tenth=<float>  fraction of peak for edge-style R [default: 0.1]

    --bin_width_phi=<float>      phi bin width (rad) for (phi,r) position histogram [default: 0.025]
    --bin_width_r=<float>        r bin width for (phi,r) position histogram [default: 0.005]
    --hist_output=<str>          path for histogram .npy; 'auto' uses prefix [default: auto]
    --hist_plot=<str>            path for histogram plot; 'auto' uses prefix; 'none' suppresses [default: auto]
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
from scipy.interpolate import RectSphereBivariateSpline as splinefit
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("args read in")
print(args)

file_str       = args['<hdf5_file>']
output_arg     = args['--output']
plot_arg       = args['--plot']
output_prefix  = args['--output_prefix']

t_start  = float(args['--t_start'])
t_end    = float(args['--t_end'])

use_cutoff     = eval(args['--use_cutoff'])
use_stddev     = eval(args['--use_stddev'])
r_cutoff_given = args['--r_cutoff'] != 'None'
if r_cutoff_given:
    r_cutoff = float(args['--r_cutoff'])

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

rho_max_fit    = float(args['--rho_max_fit'])
rho_min_fit    = float(args['--rho_min_fit'])
n_rho_fit      = int(args['--n_rho_fit'])
n_alpha_fit    = int(args['--n_alpha_fit'])
R_thresh_half  = float(args['--R_threshold_half'])
R_thresh_tenth = float(args['--R_threshold_tenth'])

bin_width_phi = float(args['--bin_width_phi'])
bin_width_r   = float(args['--bin_width_r'])
hist_out_arg  = args['--hist_output']
hist_plot_arg = args['--hist_plot']

### filename parsing ###
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]
Nphi       = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr         = int(output_suffix.split('Nr_')[1].split('_')[0])
alpha_read = str_to_float(output_suffix.split('alpha_')[1].split('_')[0])
gamma_read = str_to_float(output_suffix.split('gam_')[1].split('_')[0])
eps_read   = str_to_float(output_suffix.split('eps_')[1].split('_')[0])
nu_read    = str_to_float(output_suffix.split('nu_')[1].split('_')[0])

alpha_vals = np.array((2e-3, 1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 920, 950, 1200, 1920,
                       2372, 2500, 3200))
eps_vals   = np.array([3.3e-1, 1.0, 2.0])
nu_vals    = np.array([5e-5, 8/90000, 2e-4])
alpha = float(alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))])
gamma = float(gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))])
eps   = float(eps_vals[np.argmin(np.abs(eps_vals - eps_read))])
nu    = float(nu_vals[np.argmin(np.abs(nu_vals - nu_read))])

print(f"Parsed: Nphi={Nphi}, Nr={Nr}, alpha={alpha}, gamma={gamma}, "
      f"eps={eps}, nu={nu}")

### resolve output paths ###
def auto_or(arg, default_template):
    if arg.lower() == 'auto':
        return default_template
    return arg

output_path = auto_or(output_arg, f"{output_prefix}_{output_suffix}.npy")
if plot_arg.lower() in ('none', ''):
    plot_path = None
else:
    plot_path = auto_or(plot_arg, f"{output_prefix}_{output_suffix}.png")

print(f"output .npy: {output_path}")
print(f"plot path:   {plot_path}")

hist_output_path = auto_or(hist_out_arg, f"{output_prefix}_hist_{output_suffix}.npy")
if hist_plot_arg.lower() in ('none', ''):
    hist_plot_path = None
else:
    hist_plot_path = auto_or(hist_plot_arg, f"{output_prefix}_hist_{output_suffix}.png")
print(f"hist .npy:   {hist_output_path}")
print(f"hist plot:   {hist_plot_path}")

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
vort = dist.Field(name='vort', bases=disk)

phi_1d = phi_deal[:, 0]
r_1d   = r_deal[0, :]
Nphi_deal = len(phi_1d)
Nr_deal   = len(r_1d)

phi_mesh   = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))
theta_1d   = r_to_th(r_1d)
theta_mesh = np.tile(theta_1d[np.newaxis, :], (Nphi_deal, 1))
r_mesh     = np.tile(r_1d[np.newaxis, :],     (Nphi_deal, 1))

phi_1d_wrap = np.append(phi_1d, 2*np.pi)

### spline helpers (same as v7/v12) ###
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

### R fit helper ###
def find_threshold_crossing(rho_grid, profile, target):
    """Smallest rho where |profile(rho)| < |target|, with linear sub-grid
    interp between adjacent points. NaN if no crossing in the grid."""
    p = np.abs(profile)
    t = np.abs(target)
    below = p < t
    crossings = np.where(below)[0]
    if len(crossings) == 0:
        return np.nan
    k = crossings[0]
    if k == 0:
        return 0.0
    p_a, p_b = p[k-1], p[k]
    rho_a, rho_b = rho_grid[k-1], rho_grid[k]
    if p_a == p_b:
        return rho_a
    return rho_a + (t - p_a) * (rho_b - rho_a) / (p_b - p_a)

### open file ###
f     = h5py.File(file_str, 'r')
t_all = f['tasks/vort'].dims[0]['sim_time'][:]

t_end_eff = min(t_end, float(t_all[-1]))
if t_start > t_all[-1]:
    raise ValueError(f"t_start={t_start} beyond last available time {t_all[-1]:.3f}")

ws_start = np.where(t_all <= t_start)[0][-1] if np.any(t_all <= t_start) else 0
ws_end   = np.where(t_all >= t_end_eff)[0][0]
ws       = np.arange(ws_start, ws_end + 1)
nw, tw   = len(ws), t_all[ws]
print(f"Processing {nw} writes: t={tw[0]:.3f} to t={tw[-1]:.3f}")

### r_cutoff (same convention as v12) ###
if use_cutoff and not r_cutoff_given:
    tdur    = min((t_all[-1]-t_all[0])/3, 1/alpha)
    si      = np.where(t_all >= t_all[-1]-tdur)[0][0]
    EN_tavg = np.mean(f['tasks/EN'][si:-1])
    KE_tavg = ((eps/np.pi) - nu*EN_tavg) / (2*alpha)
    u_rms   = np.sqrt(2*KE_tavg)
    if not np.isfinite(u_rms):
        KE_tavg = np.mean(f['tasks/KE'][si:-1])
        print("using KE analysis task to estimate r_cutoff")
        u_rms = np.sqrt(2*KE_tavg)
    r_cutoff = 2*(u_rms/gamma)**(1/3) if gamma > 0 else 0.9
print(f"r_cutoff = {r_cutoff:.4f}" if use_cutoff else "r_cutoff: disabled")

if use_cutoff:
    cutoff_mask  = r_mesh >= r_cutoff
    r_cutoff_idx = np.where(r_1d < r_cutoff)[0][-1]

### CPC-frame fit grid (log-spaced in rho, with rho=0 prepended for the peak) ###
# rho_fit_grid[0] = 0 gives omega_peak; the rest are log-spaced for
# resolution near the core and clean log-scale visualization.
if n_rho_fit < 2:
    raise ValueError("n_rho_fit must be >= 2 (need at least rho=0 and one more)")
log_part       = np.logspace(np.log10(rho_min_fit), np.log10(rho_max_fit), n_rho_fit - 1)
rho_fit_grid   = np.concatenate([[0.0], log_part])
alpha_fit_grid = np.linspace(0, 2*np.pi, n_alpha_fit, endpoint=False)
rho_mesh_fit, alpha_mesh_fit = np.meshgrid(rho_fit_grid, alpha_fit_grid, indexing='ij')
print(f"rho_fit_grid: rho=0 prepended + log-spaced [{rho_min_fit:.1e}, "
      f"{rho_max_fit}] ({n_rho_fit} points total)")

### (phi, r) position histogram setup ###
# bin edges span the full disk; glitch frames are excluded from accumulation
n_phi_bins      = int(np.ceil(2*np.pi / bin_width_phi))
n_r_bins        = int(np.ceil(1.0     / bin_width_r))
phi_bin_edges   = np.linspace(0, 2*np.pi, n_phi_bins + 1)
r_bin_edges     = np.linspace(0, 1.0,     n_r_bins   + 1)
phi_bin_centers = 0.5 * (phi_bin_edges[:-1] + phi_bin_edges[1:])
r_bin_centers   = 0.5 * (r_bin_edges[:-1]   + r_bin_edges[1:])
position_hist   = np.zeros((n_phi_bins, n_r_bins), dtype=np.int64)
n_frames_hist   = 0
print(f"(phi, r) hist: {n_phi_bins} x {n_r_bins} bins  "
      f"(bin_width_phi={bin_width_phi}, bin_width_r={bin_width_r})")

### state ###
vort_mus     = []
vort_stddevs = []
lat_pois     = []
lon_pois     = []
vort_maxs    = []
th_locs      = []
r_locs       = []
phi_locs     = []
glitch_flags = []

R_half_history     = np.full(nw, np.nan)
R_tenth_history    = np.full(nw, np.nan)
omega_peak_history = np.full(nw, np.nan)
vort_radial_sum    = np.zeros(n_rho_fit)
n_valid_for_avg    = 0

r_prev    = None
phi_prev  = None
vort_prev = None

rand     = np.random.RandomState(seed=10101)
prog_cad = max(1, nw // 50)

### main loop ###
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"writes loop: i={i} out of {nw}")

    vort.load_from_hdf5(f, w)
    vort_g = np.copy(vort['g'])

    mu_fit, stddev_fit = norm.fit(vort_g)
    vort_mus.append(mu_fit)
    vort_stddevs.append(stddev_fit)

    vort_g_masked = np.copy(vort_g)
    if use_cutoff:
        vort_g_masked[cutoff_mask] = 0.
    if use_stddev:
        vort_g_masked[np.abs(vort_g_masked) <= 2*stddev_fit] = 0.

    lon_poi_idx = np.where(vort_g_masked == np.max(vort_g_masked))[0][0]
    lat_poi_idx = np.where(vort_g_masked == np.max(vort_g_masked))[1][0]

    if use_cutoff and lat_poi_idx == r_cutoff_idx:
        vort_g_ref = np.copy(vort_g)
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
                vort_g_masked = vort_g_ref
                print("  refine successful")
                success = True
                break
        if not success:
            print("  refine unsuccessful")

    lat_pois.append(theta_mesh[lon_poi_idx, lat_poi_idx])
    lon_pois.append(phi_mesh[lon_poi_idx, lat_poi_idx])

    if use_interp:
        lm, la, ds, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g_masked,
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
        data_max = np.max(vort_g_masked)
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
        continue

    ### update (phi, r) position histogram (non-glitch frames only) ###
    # use the actual bin edges (np.digitize) — nominal bin_width_phi/r are
    # rounded up to fit a whole number of bins on [0, 2pi] / [0, 1.0]
    phi_idx = int(np.digitize(lon_loc % (2*np.pi), phi_bin_edges)) - 1
    r_idx   = int(np.digitize(r_loc,               r_bin_edges))   - 1
    phi_idx = max(0, min(phi_idx, n_phi_bins - 1))
    r_idx   = max(0, min(r_idx,   n_r_bins   - 1))
    if 0 <= r_loc <= 1.0:  # only accumulate physically valid r
        position_hist[phi_idx, r_idx] += 1
        n_frames_hist += 1

    ### CPC-frame radial vorticity profile (alpha-average) for R fit ###
    x_CPC_t = r_loc * np.cos(lon_loc)
    y_CPC_t = r_loc * np.sin(lon_loc)
    x_lab   = x_CPC_t + rho_mesh_fit * np.cos(alpha_mesh_fit)
    y_lab   = y_CPC_t + rho_mesh_fit * np.sin(alpha_mesh_fit)
    r_lab   = np.sqrt(x_lab**2 + y_lab**2)
    phi_lab = np.arctan2(y_lab, x_lab) % (2*np.pi)
    valid   = r_lab <= r_1d[-1]

    vort_g_wrap = np.vstack([vort_g, vort_g[0:1, :]])
    interp_vort = RegularGridInterpolator(
        (phi_1d_wrap, r_1d), vort_g_wrap,
        method='linear', bounds_error=False, fill_value=np.nan)
    pts_flat = np.column_stack([phi_lab.ravel(), r_lab.ravel()])
    vort_cpc = interp_vort(pts_flat).reshape(n_rho_fit, n_alpha_fit)

    vort_radial = np.zeros(n_rho_fit)
    for k in range(n_rho_fit):
        vmask = valid[k, :] & np.isfinite(vort_cpc[k, :])
        if np.sum(vmask) > 0:
            vort_radial[k] = np.mean(vort_cpc[k, vmask])
        else:
            vort_radial[k] = np.nan

    if np.all(np.isfinite(vort_radial)):
        vort_radial_sum += vort_radial
        n_valid_for_avg += 1

    omega_0 = vort_radial[0]
    omega_peak_history[i] = omega_0

    R_half_history[i]  = find_threshold_crossing(rho_fit_grid, vort_radial,
                                                  R_thresh_half  * omega_0)
    R_tenth_history[i] = find_threshold_crossing(rho_fit_grid, vort_radial,
                                                  R_thresh_tenth * omega_0)

f.close()

### summary stats ###
vort_radial_tavg = vort_radial_sum / max(n_valid_for_avg, 1)

def safe_stats(arr):
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(finite)), float(np.std(finite)), float(np.median(finite))

R_half_mean,  R_half_std,  R_half_median  = safe_stats(R_half_history)
R_tenth_mean, R_tenth_std, R_tenth_median = safe_stats(R_tenth_history)

print(f"\n{'='*60}")
print(f"R summary  (n_valid frames = {n_valid_for_avg} of {nw})")
print(f"{'='*60}")
print(f"  R_half  (peak fraction = {R_thresh_half}):  "
      f"mean = {R_half_mean:.4f},  median = {R_half_median:.4f},  "
      f"std = {R_half_std:.4f}")
print(f"  R_tenth (peak fraction = {R_thresh_tenth}): "
      f"mean = {R_tenth_mean:.4f},  median = {R_tenth_median:.4f},  "
      f"std = {R_tenth_std:.4f}")
n_glitch = int(np.sum(glitch_flags))
print(f"  glitches: {n_glitch} of {nw}")

### save ###
processed = {
    # tracking
    'nw': nw, 'ws': ws, 'tw': tw,
    'r_deal': r_deal, 'phi_deal': phi_deal,
    'r_1d': r_1d, 'phi_1d': phi_1d,
    'vort_mus': vort_mus, 'vort_stddevs': vort_stddevs,
    'lat_pois': lat_pois, 'lon_pois': lon_pois,
    'vort_maxs': vort_maxs,
    'th_locs': th_locs, 'r_locs': r_locs, 'phi_locs': phi_locs,
    'glitch_flags': glitch_flags,
    # R fit
    'rho_fit_grid'      : rho_fit_grid,
    'alpha_fit_grid'    : alpha_fit_grid,
    'R_half_history'    : R_half_history,
    'R_tenth_history'   : R_tenth_history,
    'omega_peak_history': omega_peak_history,
    'vort_radial_tavg'  : vort_radial_tavg,
    'n_valid_for_avg'   : n_valid_for_avg,
    'R_thresh_half'     : R_thresh_half,
    'R_thresh_tenth'    : R_thresh_tenth,
    'R_half_mean'       : R_half_mean,   'R_half_std'    : R_half_std,    'R_half_median'  : R_half_median,
    'R_tenth_mean'      : R_tenth_mean,  'R_tenth_std'   : R_tenth_std,   'R_tenth_median' : R_tenth_median,
    # metadata
    'gamma': gamma, 'alpha': alpha, 'eps': eps, 'nu': nu,
    'Nphi': Nphi, 'Nr': Nr,
    'r_cutoff': r_cutoff if use_cutoff else None,
    'output_suffix': output_suffix,
}
print(f"\nSaving: {output_path}")
np.save(output_path, processed)

### save (phi, r) position histogram (separate file) ###
hist_processed = {
    'position_hist'  : position_hist,
    'phi_bin_edges'  : phi_bin_edges,
    'r_bin_edges'    : r_bin_edges,
    'phi_bin_centers': phi_bin_centers,
    'r_bin_centers'  : r_bin_centers,
    'bin_width_phi'  : bin_width_phi,
    'bin_width_r'    : bin_width_r,
    'n_phi_bins'     : n_phi_bins,
    'n_r_bins'       : n_r_bins,
    'n_frames_hist'  : n_frames_hist,
    'tw'             : tw,
    # metadata
    'gamma': gamma, 'alpha': alpha, 'eps': eps, 'nu': nu,
    'Nphi' : Nphi,  'Nr'   : Nr,
    'output_suffix': output_suffix,
}
print(f"Saving histogram: {hist_output_path}  "
      f"({n_frames_hist} frames accumulated)")
np.save(hist_output_path, hist_processed)

### diagnostic plot ###
if plot_path is not None:
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # panel A: trajectory r(t) and r-phi scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tw, r_locs, color='C0', lw=0.8, label=r'$r_\mathrm{CPC}(t)$')
    glitch_arr = np.array(glitch_flags, dtype=bool)
    if np.any(glitch_arr):
        ax.scatter(tw[glitch_arr], np.array(r_locs)[glitch_arr],
                   color='red', s=10, label='glitches')
    ax.set_xlabel('t')
    ax.set_ylabel(r'$r_\mathrm{CPC}$')
    ax.set_title(f'CPC trajectory — $\\gamma={gamma:.0f}$, {nw} frames')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1], projection='polar')
    ax.scatter(phi_locs, r_locs, s=3, c=tw, cmap='viridis', alpha=0.6)
    ax.set_title('CPC position (polar)', fontsize=10)
    ax.set_ylim([0, max(0.8, float(np.nanmax(r_locs)) * 1.1)])

    # panel B: R histograms
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, rho_max_fit, 40)
    ax.hist(R_half_history[np.isfinite(R_half_history)],   bins=bins,
            alpha=0.55, color='C0', label=f'$R_{{1/2}}$ (mean={R_half_mean:.3f})')
    ax.hist(R_tenth_history[np.isfinite(R_tenth_history)], bins=bins,
            alpha=0.55, color='C3', label=f'$R_{{1/10}}$ (mean={R_tenth_mean:.3f})')
    ax.axvline(R_half_mean,  color='C0', ls='--', lw=1.0)
    ax.axvline(R_tenth_mean, color='C3', ls='--', lw=1.0)
    ax.set_xlabel(r'$R$')
    ax.set_ylabel('frames')
    ax.set_title(r'Histograms of $R$ thresholds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # panel C: R(t)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(tw, R_half_history,  color='C0', lw=0.8, label=r'$R_{1/2}(t)$')
    ax.plot(tw, R_tenth_history, color='C3', lw=0.8, label=r'$R_{1/10}(t)$')
    ax.axhline(R_half_mean,  color='C0', ls='--', lw=1.0, alpha=0.7)
    ax.axhline(R_tenth_mean, color='C3', ls='--', lw=1.0, alpha=0.7)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$R$')
    ax.set_title(r'$R(t)$ time series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # panel D: time-averaged radial vorticity profile + thresholds (log rho)
    ax = fig.add_subplot(gs[1, 2])
    profile  = vort_radial_tavg
    peak     = profile[0]   # rho=0
    # skip rho=0 for log-axis plotting; show peak as annotation/marker
    ax.plot(rho_fit_grid[1:], profile[1:], color='C2', lw=1.5,
            label=r'$\bar\omega(\rho)$ (time-avg)')
    ax.scatter([rho_fit_grid[1]], [peak], color='C2', marker='*', s=90, zorder=5,
               label=f'peak (at $\\rho=0$) = {peak:.2f}')
    ax.axhline(R_thresh_half  * peak, color='C0', ls=':', lw=0.8,
               label=f'{R_thresh_half:.0%} of peak')
    ax.axhline(R_thresh_tenth * peak, color='C3', ls=':', lw=0.8,
               label=f'{R_thresh_tenth:.0%} of peak')
    ax.axvline(R_half_mean,  color='C0', ls='--', lw=1.0,
               label=f'$\\bar R_{{1/2}}={R_half_mean:.3f}$')
    ax.axvline(R_tenth_mean, color='C3', ls='--', lw=1.0,
               label=f'$\\bar R_{{1/10}}={R_tenth_mean:.3f}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\rho$')
    ax.set_ylabel(r'$\bar\omega$ (CPC frame, $\alpha$-avg, $t$-avg)')
    ax.set_title('Time-averaged radial vorticity profile')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'Locator + R fit  —  {output_suffix}', fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

### dedicated histogram plot: 1D r-hist, 1D phi-hist, 2D polar residency map ###
if hist_plot_path is not None:
    glitch_arr = np.array(glitch_flags, dtype=bool)
    r_arr_ok   = np.array(r_locs)[~glitch_arr]
    phi_arr_ok = np.array(phi_locs)[~glitch_arr]
    r_mean_ok    = float(np.nanmean(r_arr_ok))   if len(r_arr_ok)   > 0 else np.nan
    r_median_ok  = float(np.nanmedian(r_arr_ok)) if len(r_arr_ok)   > 0 else np.nan
    # circular mean for phi
    if len(phi_arr_ok) > 0:
        phi_mean_ok = float(np.arctan2(np.mean(np.sin(phi_arr_ok)),
                                        np.mean(np.cos(phi_arr_ok))) % (2*np.pi))
    else:
        phi_mean_ok = np.nan

    # 1D marginals from the 2D histogram (consistent by construction)
    r_marg   = position_hist.sum(axis=0)  # shape (n_r_bins,)
    phi_marg = position_hist.sum(axis=1)  # shape (n_phi_bins,)

    # outer r for plot zoom (a bit beyond the support)
    nz_r_idx = np.where(r_marg > 0)[0]
    if len(nz_r_idx) > 0:
        r_outer = min(1.0, r_bin_edges[nz_r_idx[-1] + 1] * 1.2)
    else:
        r_outer = 1.0

    fig_h = plt.figure(figsize=(13.5, 6.5), constrained_layout=True)
    gs_h  = fig_h.add_gridspec(2, 3)

    # panel: 1D r-histogram
    ax1 = fig_h.add_subplot(gs_h[0, 0])
    ax1.bar(r_bin_centers, r_marg, width=r_bin_edges[1] - r_bin_edges[0],
            align='center', color='C0', alpha=0.85)
    if np.isfinite(r_mean_ok):
        ax1.axvline(r_mean_ok,   color='k',   ls='--', lw=1.0,
                    label=fr'$\bar r={r_mean_ok:.3f}$')
    if np.isfinite(r_median_ok):
        ax1.axvline(r_median_ok, color='gray', ls=':',  lw=1.0,
                    label=fr'median $r={r_median_ok:.3f}$')
    ax1.set_xlim([0, r_outer])
    ax1.set_xlabel(r'$r$')
    ax1.set_ylabel('count')
    ax1.set_title('1D $r$ histogram')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # panel: 1D phi-histogram
    ax2 = fig_h.add_subplot(gs_h[1, 0])
    ax2.bar(phi_bin_centers, phi_marg,
            width=phi_bin_edges[1] - phi_bin_edges[0],
            align='center', color='C1', alpha=0.85)
    if np.isfinite(phi_mean_ok):
        ax2.axvline(phi_mean_ok, color='k', ls='--', lw=1.0,
                    label=fr'$\bar\varphi={phi_mean_ok:.3f}$ rad')
    ax2.set_xlim([0, 2*np.pi])
    ax2.set_xticks(np.arange(0, 2*np.pi + 0.01, np.pi/2))
    ax2.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax2.set_xlabel(r'$\varphi$')
    ax2.set_ylabel('count')
    ax2.set_title(r'1D $\varphi$ histogram')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # panel: 2D polar residency (spans both rows)
    ax3 = fig_h.add_subplot(gs_h[0:2, 1:3], projection='polar')
    phi_mesh_hist, r_mesh_hist = np.meshgrid(phi_bin_edges, r_bin_edges,
                                              indexing='ij')
    # explicit vmin=0 so empty bins map to colormap floor;
    # use shading='flat' explicitly (edges -> faces interpretation)
    pcm = ax3.pcolormesh(phi_mesh_hist, r_mesh_hist,
                          np.log10(position_hist + 1),
                          shading='flat', cmap='magma',
                          vmin=0, vmax=np.log10(position_hist.max() + 1))
    if np.isfinite(r_mean_ok) and np.isfinite(phi_mean_ok):
        ax3.plot([phi_mean_ok], [r_mean_ok], 'cx', ms=14, mew=2.0,
                 label=fr'mean: $\bar r={r_mean_ok:.3f}$, $\bar\varphi={phi_mean_ok:.3f}$')
    ax3.set_ylim([0, r_outer])
    ax3.set_title(fr'2D residency (polar)  —  $\gamma={gamma:.0f}$, '
                  fr'$n_\mathrm{{frames}}={n_frames_hist}$', fontsize=10)
    ax3.legend(loc='lower right', fontsize=8)
    fig_h.colorbar(pcm, ax=ax3, label=r'$\log_{10}(\mathrm{count}+1)$',
                    fraction=0.046, pad=0.08)

    fig_h.suptitle(f'Position histograms  —  {output_suffix}', fontsize=11)
    fig_h.savefig(hist_plot_path, dpi=130)
    plt.close(fig_h)
    print(f"Histogram figure saved: {hist_plot_path}")

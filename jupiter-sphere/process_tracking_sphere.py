"""
Processing script for locating dominant polar vortices and estimating
their scales for Dedalus simulations of forced-dissipative turbulence
solved on the surface of a sphere.

--- Description ---

For each pole (north at colatitude theta=0, south at theta=pi), the
script locates a single (C)PC ((circum)polar cyclone) as the grid point
with the largest |vort| inside a polar cap of angular radius cap_colat,
then characterizes its size by alpha-averaging the vorticity (i.e., averaging
azimuthally in a frame centered on the CPC) and finding the angular radii at which
the alpha-averaged profile crosses fractions of the central peak vorticity
(R_half, R_fifth, R_tenth) or the global rms vorticity (R_rms).

All outputs are saved in a dict object in a single .npy file at output_path.
The most useful keys, organized by topic:

  Per-frame tracking arrays  (pole index: 0 = N, 1 = S):
    theta_locs[pole, frame]        - CPC colatitude (rad)
    phi_locs[pole, frame]          - CPC longitude (rad)
    vort_peak_history[pole, frame] - signed vort at CPC center
    sign_history[pole, frame]      - +/-1 sign of vort_peak
    tw[frame]                      - sim time of each frame

  CPC length scales  (R_type index: 0 = R_half, 1 = R_fifth,
                                    2 = R_tenth, 3 = R_rms):
    R_history[pole, R_type, frame] - per-frame R (rad); NaN if no crossing
    R_stats[pole, R_type, stat]    - summary stats; 0 = mean, 1 = std
    R_labels                       - list naming the R_type axis
    R_threshold_fractions          - fractions used for R_half / fifth / tenth

  CPC-frame radial profile and grids:
    vort_radial_tavg[pole, rho_idx], rho_fit_grid[rho_idx]

  Whole-sphere vorticity rms:
    vort_rms_history[frame], vort_rms_tavg

  Lab-frame position residency (2D histogram in (phi, theta)):
    position_hist[pole, phi_bin, theta_bin]
    phi_bin_edges, theta_bin_edges, n_frames_hist[pole]

  Antipodality diagnostics (each is shape (frame,)):
    antipodal_gc_dist      - great-circle distance between CPC_N and the
                             antipode of CPC_S (zero if perfectly antipodal)
    antipodal_delta_theta  - theta_N + theta_S - pi  (also = the diff in
                             each CPC's colat. separation from its own pole)
    antipodal_delta_phi    - wrap(phi_N - phi_S - pi)  (zero if antipodal)
    relative_zonal_phase   - wrap(phi_N - phi_S)  (pi if antipodal)
    antipodal_stats[metric, stat] - mean/std per metric;
                             metrics 0..3 in order above

  Convention notes:
    Colatitude theta in [0, pi] with theta=0 at "north" (Dedalus standard).
    Pole index 0 = small theta (N), pole index 1 = large theta (S).

--- Usage notes ---
Usage:
    process_tracking_locator_sphere_v1.py <hdf5_file> [options]

If the simulation includes multiple snapshot files (snapshots_s1.h5,
snapshots_s2.h5, ...), merge them into a single file by running, inside
the run directory:
    from dedalus.tools.post import merge_analysis
    merge_analysis('snapshots', cleanup=False)
That produces 'snapshots/snapshots.h5' which is what to pass below.

Arguments:
    <hdf5_file>   merged HDF5 snapshot file

Options:
    --output=<str>           output .npy path; 'auto' uses output_suffix [default: auto]
    --plot=<str>             plot path; 'auto'; 'none' to skip [default: auto]
    --output_prefix=<str>    prefix used when paths are 'auto' [default: processed_tracking_locator_sphere]
    --output_suffix=<str>    suffix; 'auto' derives from filename [default: auto]

    --t_start=<float>        sim time to begin tracking [default: 0.]
    --t_end=<float>          sim time to stop tracking  [default: 1e9]

    --vort_task=<str>        HDF5 task name for vorticity [default: vorticity]

    --gamma=<str>            gamma (= 2*Omega); 'auto' uses 2*1186 [default: auto]
    --alpha=<float>          friction [default: 0.0333333]
    --eps=<float>            energy injection rate [default: 1.0]

    --cap_colat=<str>        outer colatitude of each polar cap (rad), clamped
                             to <= pi/2; 'auto' uses 2 * (spherical Rhines
                             angle) + 0.05 for gamma>0, or pi/2 (hemispheric)
                             for gamma=0 [default: auto]

    --rho_max_fit=<float>    outer rho for the CPC-frame fit grid (rad) [default: 0.4]
    --rho_min_fit=<float>    smallest non-zero rho on log-spaced grid [default: 1e-3]
    --n_rho_fit=<int>        number of rho points (rho=0 prepended) [default: 96]
    --n_alpha_fit=<int>      number of alpha points (uniform) [default: 32]
    --R_threshold_half=<float>   [default: 0.5]
    --R_threshold_fifth=<float>  [default: 0.2]
    --R_threshold_tenth=<float>  [default: 0.1]

    --bin_width_phi=<float>      phi bin width (rad) [default: 0.05]
    --bin_width_theta=<float>    theta bin width (rad) [default: 0.005]
"""

### --- Beginning of organizational tasks --- ###

### imports ###
import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3
from docopt import docopt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

### parse args ###
args = docopt(__doc__)
print(args)

file_str       = args['<hdf5_file>']
output_arg     = args['--output']
plot_arg       = args['--plot']
output_prefix  = args['--output_prefix']
output_suffix_arg = args['--output_suffix']
t_start        = float(args['--t_start'])
t_end          = float(args['--t_end'])
vort_task      = args['--vort_task']
gamma_arg      = args['--gamma']
alpha          = float(args['--alpha'])
eps            = float(args['--eps'])
cap_colat_arg  = args['--cap_colat']
rho_max_fit    = float(args['--rho_max_fit'])
rho_min_fit    = float(args['--rho_min_fit'])
n_rho_fit      = int(args['--n_rho_fit'])
n_alpha_fit    = int(args['--n_alpha_fit'])
R_thresh_half  = float(args['--R_threshold_half'])
R_thresh_fifth = float(args['--R_threshold_fifth'])
R_thresh_tenth = float(args['--R_threshold_tenth'])
bin_width_phi   = float(args['--bin_width_phi'])
bin_width_theta = float(args['--bin_width_theta'])

### determine gamma from Omega (the rotation rate of the IVP) ###
if gamma_arg.lower() == 'auto':
    Omega_default = 1186.0  # matches rotating_script.py
    gamma = 2 * Omega_default
else:
    gamma = float(gamma_arg)
print(f"gamma = {gamma:.4f}, alpha = {alpha:.4f}, eps = {eps:.4f}")

### output paths ###
def extract_output_suffix(file_path):
    basename = file_path.split('/')[-1]
    for ext in ('.h5', '.hdf5'):
        if basename.endswith(ext):
            basename = basename[:-len(ext)]; break
    for prefix in ('snapshots_', 'snapshots'):
        if basename.startswith(prefix):
            basename = basename[len(prefix):].lstrip('_'); break
    return basename if basename else 'sphere'

if output_suffix_arg.lower() == 'auto':
    output_suffix = extract_output_suffix(file_str)
else:
    output_suffix = output_suffix_arg
print(f"output_suffix: {output_suffix}")

# Each path argument can be 'auto' (build a default from the suffix),
# 'none' (skip producing that output), or a concrete file path.
def path_or_default(arg, default_path):
    """Return arg if it's a concrete path, or the default if arg is 'auto'."""
    return default_path if arg.lower() == 'auto' else arg

output_path = path_or_default(output_arg, f"{output_prefix}_{output_suffix}.npy")
plot_path = None if plot_arg.lower() in ('none', '') else \
    path_or_default(plot_arg, f"{output_prefix}_{output_suffix}.png")
print(f"output .npy: {output_path}")
print(f"plot path:   {plot_path}")

### --- End of organizational tasks --- ###

### cap outer colatitude (where to cut off the polar search) ###
# The Rhines latitude on the sphere is the angular distance theta* from
# the pole at which the local Rhines length matches theta* itself.  Using
# the local meridional gradient of f = 2*Omega*cos(theta), which is
# beta(theta) = 2*Omega*sin(theta), the local Rhines length is
# sqrt(U/beta), so theta* satisfies
#     theta*^2 * sin(theta*) = U / (2*Omega) = U / gamma.
# In the small-angle limit this reduces to the flat-plane (U/gamma)^(1/3).
if cap_colat_arg.lower() == 'auto':
    U_rms = np.sqrt(eps / alpha)
    if gamma <= 0:
        # No rotational symmetry breaking: use hemispherical caps so the
        # CPCs can roam freely.  Note: if both CPCs land in the same
        # hemisphere, the script will still record a "second" extremum in
        # the empty hemisphere -- the user can spot this by comparing
        # |vort_peak| between the two recorded poles.
        L_rh      = np.nan
        cap_colat = np.pi / 2
        print(f"cap_colat (auto, gamma=0): hemispheric, "
              f"cap_colat=pi/2 ({cap_colat:.4f})")
    else:
        C = U_rms / gamma
        L_rh_flat = C**(1/3)  # flat-plane / small-angle limit
        rhines_eq = lambda theta: theta**2 * np.sin(theta) - C
        if rhines_eq(np.pi / 2) < 0:
            # the Rhines latitude would be past the equator; clamp there
            L_rh = np.pi / 2
        else:
            L_rh = float(brentq(rhines_eq, 1e-6, np.pi / 2))
        # cap is 2*L_rh + 0.05 but no more than pi/2 (hemispheric upper bound)
        cap_colat = float(min(2 * L_rh + 0.05, np.pi / 2))
        print(f"cap_colat (auto): U_rms={U_rms:.4f}, "
              f"L_rh_flat={L_rh_flat:.4f}, L_rh_sphere={L_rh:.4f}, "
              f"cap_colat={cap_colat:.4f}")
else:
    cap_colat = float(cap_colat_arg)
    if cap_colat > np.pi / 2:
        cap_colat = np.pi / 2
        print(f"  warning: cap_colat capped at pi/2 (hemispheric upper bound)")
    print(f"cap_colat (CLI): {cap_colat:.4f}")

### open HDF5 ###
f = h5py.File(file_str, 'r')
if f"tasks/{vort_task}" not in f:
    available = list(f.get('tasks', {}).keys())
    raise KeyError(f"Task '{vort_task}' not in HDF5. Available: {available}")
vort_dset = f[f'tasks/{vort_task}']
t_all     = vort_dset.dims[0]['sim_time'][:]
Nphi_grid, Ntheta_grid = vort_dset.shape[1], vort_dset.shape[2]
print(f"HDF5: {len(t_all)} writes, t in [{t_all[0]:.3f}, {t_all[-1]:.3f}], "
      f"grid ({Nphi_grid} phi, {Ntheta_grid} theta)")

### Dedalus basis ###
coords = d3.S2Coordinates('phi', 'theta')
dist   = d3.Distributor(coords, dtype=np.float64)
full_basis = d3.SphereBasis(coords, (Nphi_grid, Ntheta_grid),
                             radius=1.0, dealias=3/2, dtype=np.float64)
phi_grid_d3, theta_grid_d3 = dist.local_grids(full_basis, scales=(1, 1))
# Dedalus convention: phi runs 0 -> 2*pi (exclusive), theta descends from
# pi (exclusive) to 0 (exclusive).  We flip theta to ascending (N -> S)
# for all local processing in this script.
phi_1d      = phi_grid_d3[:, 0]
theta_1d    = np.flip(theta_grid_d3[0, :])
phi_1d_wrap = np.append(phi_1d, 2 * np.pi)  # for interpolation across phi=0

# Reusable field used inside the main loop to take sphere averages.
vort_field = dist.Field(name='vort', bases=full_basis)

# cap masks (on the ascending theta_1d).  At hemispheric cap_colat = pi/2,
# use strict inequalities so a grid point sitting exactly on the equator
# wouldn't get claimed by both caps and risk double-counting an extremum.
n_poles = 2
if cap_colat >= np.pi / 2 - 1e-12:
    cap_idxs_per_pole = [
        np.where(theta_1d < np.pi / 2)[0],   # 0: N, theta < pi/2
        np.where(theta_1d > np.pi / 2)[0],   # 1: S, theta > pi/2
    ]
else:
    cap_idxs_per_pole = [
        np.where(theta_1d <= cap_colat)[0],
        np.where(theta_1d >= np.pi - cap_colat)[0],
    ]
print(f"cap sizes:  N={len(cap_idxs_per_pole[0])}, "
      f"S={len(cap_idxs_per_pole[1])}")

### selection of time window for tracking ###
t_end_eff = min(t_end, float(t_all[-1]))
if t_start > t_all[-1]:
    raise ValueError(f"t_start={t_start} > last available time {t_all[-1]:.3f}")
ws_start = np.where(t_all <= t_start)[0][-1] if np.any(t_all <= t_start) else 0
ws_end   = np.where(t_all >= t_end_eff)[0][0]
ws       = np.arange(ws_start, ws_end + 1)
nw, tw   = len(ws), t_all[ws]
print(f"processing {nw} writes: t={tw[0]:.3f} to t={tw[-1]:.3f}")

### CPC-frame fit grid and histogram bins ###
if n_rho_fit < 2:
    raise ValueError("n_rho_fit must be >= 2")
rho_fit_grid = np.concatenate([
    [0.0],
    np.logspace(np.log10(rho_min_fit), np.log10(rho_max_fit), n_rho_fit - 1),
])
alpha_fit_grid = np.linspace(0, 2 * np.pi, n_alpha_fit, endpoint=False)

n_phi_bins      = int(np.ceil(2 * np.pi / bin_width_phi))
n_theta_bins    = int(np.ceil(np.pi     / bin_width_theta))
phi_bin_edges   = np.linspace(0, 2 * np.pi, n_phi_bins   + 1)
theta_bin_edges = np.linspace(0, np.pi,     n_theta_bins + 1)
phi_bin_centers   = 0.5 * (phi_bin_edges[:-1]   + phi_bin_edges[1:])
theta_bin_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])

### helper functions ###
def great_circle_distance(theta1, phi1, theta2, phi2):
    """Great-circle distance between two points on the unit sphere (rad)."""
    cos_d = (np.cos(theta1) * np.cos(theta2)
           + np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    return np.arccos(np.clip(cos_d, -1.0, 1.0))

def find_max_abs_in_cap(vort_field_g, cap_idxs):
    """Locate the grid point with the largest |vort| inside the cap.
    Returns (phi_idx, theta_idx_global, vort_value, sign)."""
    sub = vort_field_g[:, cap_idxs]
    flat = int(np.argmax(np.abs(sub)))
    phi_idx, theta_idx_in_cap = np.unravel_index(flat, sub.shape)
    theta_idx_global = int(cap_idxs[theta_idx_in_cap])
    val = float(vort_field_g[phi_idx, theta_idx_global])
    return int(phi_idx), theta_idx_global, val, int(np.sign(val))

def find_threshold_crossing(rho_grid, profile, target):
    """Smallest rho where |profile| drops below |target|, with sub-grid
    linear interp between adjacent points. NaN if no crossing."""
    prof, targ = np.abs(profile), np.abs(target)
    crossings = np.where(prof < targ)[0]
    if len(crossings) == 0 or crossings[0] == 0:
        return np.nan
    k = crossings[0]
    return rho_grid[k - 1] + (targ - prof[k - 1]) * (rho_grid[k] - rho_grid[k - 1]) / (prof[k] - prof[k - 1])

def cpc_frame_to_lab(theta_cpc, phi_cpc, rho_grid, alpha_grid):
    """
    Map CPC-frame (rho_gc, alpha) coords to lab-frame (theta, phi) on the
    sphere.  rho_gc is great-circle distance from the CPC center; alpha
    is azimuthal angle measured counter-clockwise from local east (viewed
    from outside the sphere).
    """
    # 2D meshes of CPC-frame coords; shape (n_rho, n_alpha)
    rho_md, alpha_md = np.meshgrid(rho_grid, alpha_grid, indexing='ij')

    # Orthonormal CPC-centered basis (each row is a 3-component vector):
    #   c_hat = outward normal at CPC,  e_hat = local east,  n_hat = local north
    c_hat = np.array([np.sin(theta_cpc) * np.cos(phi_cpc),
                      np.sin(theta_cpc) * np.sin(phi_cpc),
                      np.cos(theta_cpc)])
    e_hat = np.array([-np.sin(phi_cpc), np.cos(phi_cpc), 0.0])
    n_hat = np.array([-np.cos(theta_cpc) * np.cos(phi_cpc),
                      -np.cos(theta_cpc) * np.sin(phi_cpc),
                       np.sin(theta_cpc)])

    # The trailing length-1 axis from [:, :, None] lines up against the
    # length-3 axis of each basis vector during broadcasting, producing
    # the (n_rho, n_alpha, 3) output below.
    cos_rho   = np.cos(rho_md)[:, :, None]
    sin_rho   = np.sin(rho_md)[:, :, None]
    cos_alpha = np.cos(alpha_md)[:, :, None]
    sin_alpha = np.sin(alpha_md)[:, :, None]
    p_hat = (cos_rho * c_hat
           + sin_rho * (cos_alpha * e_hat + sin_alpha * n_hat))
    theta_lab = np.arccos(np.clip(p_hat[:, :, 2], -1.0, 1.0))
    phi_lab   = np.arctan2(p_hat[:, :, 1], p_hat[:, :, 0]) % (2 * np.pi)
    return theta_lab, phi_lab

### pre-allocate storage ###
theta_locs        = np.full((n_poles, nw), np.nan)
phi_locs          = np.full((n_poles, nw), np.nan)
vort_peak_history = np.full((n_poles, nw), np.nan)
sign_history      = np.zeros((n_poles, nw), dtype=int)

# R_history axis ordering: (pole, R_type, frame)
# R_type 0..3 = R_half, R_fifth, R_tenth, R_rms
R_labels        = ['R_half', 'R_fifth', 'R_tenth', 'R_rms']
n_R_types       = len(R_labels)
peak_thresholds = np.array([R_thresh_half, R_thresh_fifth, R_thresh_tenth])
R_history       = np.full((n_poles, n_R_types, nw), np.nan)

vort_radial_sum  = np.zeros((n_poles, n_rho_fit))
n_valid_for_avg  = np.zeros(n_poles, dtype=int)
position_hist    = np.zeros((n_poles, n_phi_bins, n_theta_bins), dtype=np.int64)
n_frames_hist    = np.zeros(n_poles, dtype=int)
vort_rms_history = np.full(nw, np.nan)

### --- main loop --- ###
prog_cad = max(1, nw // 50)
print(f"\nStarting per-frame locator (nw={nw}) ...")
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")

    # raw vort_dset[w] is in Dedalus order (descending theta);
    # set the Dedalus field for the sphere average, then flip to ascending
    # for everything else.
    vort_raw = np.array(vort_dset[w])
    vort_field['g'] = vort_raw
    vort_rms_t = float(np.sqrt(
        max(float(d3.Average(vort_field * vort_field).evaluate()['g'].flat[0]),
            0.0)))
    vort_rms_history[i] = vort_rms_t

    vort_g = np.flip(vort_raw, axis=1)  # ascending theta to match theta_1d

    # build the phi-wrapped interpolator once per frame (reused per pole)
    vort_wrap = np.vstack([vort_g, vort_g[0:1, :]])
    interp_vort = RegularGridInterpolator(
        (phi_1d_wrap, theta_1d), vort_wrap,
        method='linear', bounds_error=False, fill_value=np.nan)

    for p in range(n_poles):
        phi_idx, theta_idx, vort_val, sign = find_max_abs_in_cap(
            vort_g, cap_idxs_per_pole[p])
        theta_p, phi_p = theta_1d[theta_idx], phi_1d[phi_idx]
        theta_locs[p, i]        = theta_p
        phi_locs[p, i]          = phi_p
        vort_peak_history[p, i] = vort_val
        sign_history[p, i]      = sign

        # CPC-frame alpha-averaged radial profile
        theta_lab, phi_lab = cpc_frame_to_lab(
            theta_p, phi_p, rho_fit_grid, alpha_fit_grid)
        pts      = np.column_stack([phi_lab.ravel(), theta_lab.ravel()])
        vort_cpc = interp_vort(pts).reshape(theta_lab.shape)
        vort_radial = np.nanmean(vort_cpc, axis=1)

        # length scales:  R_half / R_fifth / R_tenth = fractions of peak;
        # R_rms = where profile drops past sign * vort_rms
        peak_val = vort_radial[0]
        if np.isfinite(peak_val) and abs(peak_val) > 0:
            for k, frac in enumerate(peak_thresholds):
                R_history[p, k, i] = find_threshold_crossing(
                    rho_fit_grid, vort_radial, frac * peak_val)
            R_history[p, 3, i] = find_threshold_crossing(
                rho_fit_grid, vort_radial, sign * vort_rms_t)
            if np.all(np.isfinite(vort_radial)):
                vort_radial_sum[p] += vort_radial
                n_valid_for_avg[p] += 1

        # 2D position histogram
        phi_bin   = max(0, min(
            int(np.digitize(phi_p % (2 * np.pi), phi_bin_edges)) - 1,
            n_phi_bins - 1))
        theta_bin = max(0, min(
            int(np.digitize(theta_p, theta_bin_edges)) - 1,
            n_theta_bins - 1))
        position_hist[p, phi_bin, theta_bin] += 1
        n_frames_hist[p] += 1

f.close()
print("Per-frame loop done.")

### --- post-loop computations --- ###
vort_radial_tavg = vort_radial_sum / np.maximum(n_valid_for_avg[:, None], 1)
vort_rms_tavg    = float(np.nanmean(vort_rms_history))

# R_stats[pole, R_type, stat]: stat 0 = mean, 1 = std
R_stats = np.full((n_poles, n_R_types, 2), np.nan)
for p in range(n_poles):
    for k in range(n_R_types):
        fin = R_history[p, k, :][np.isfinite(R_history[p, k, :])]
        if len(fin) > 0:
            R_stats[p, k, 0] = float(np.mean(fin))
            R_stats[p, k, 1] = float(np.std(fin))

# antipodality
theta_N, theta_S = theta_locs[0, :], theta_locs[1, :]
phi_N,   phi_S   = phi_locs[0, :],   phi_locs[1, :]
theta_S_antipode = np.pi - theta_S
phi_S_antipode   = (phi_S + np.pi) % (2 * np.pi)
antipodal_gc_dist     = great_circle_distance(
    theta_N, phi_N, theta_S_antipode, phi_S_antipode)
antipodal_delta_theta = theta_N + theta_S - np.pi
antipodal_delta_phi   = np.arctan2(
    np.sin(phi_N - phi_S - np.pi), np.cos(phi_N - phi_S - np.pi))
relative_zonal_phase  = np.arctan2(
    np.sin(phi_N - phi_S), np.cos(phi_N - phi_S))

antipodal_stats = np.array([
    [float(np.mean(antipodal_gc_dist)),     float(np.std(antipodal_gc_dist))],
    [float(np.mean(antipodal_delta_theta)), float(np.std(antipodal_delta_theta))],
    [float(np.mean(antipodal_delta_phi)),   float(np.std(antipodal_delta_phi))],
    [float(np.mean(relative_zonal_phase)),  float(np.std(relative_zonal_phase))],
])
antipodal_stat_labels = ['gc_dist', 'delta_theta', 'delta_phi',
                          'relative_zonal_phase']

### print summary ###
print(f"\n{'='*60}\nSummary\n{'='*60}")
for p, name in enumerate(['N', 'S']):
    print(f"  {name} pole: <theta>={float(np.nanmean(theta_locs[p])):.4f},  "
          f"<|vort_peak|>={float(np.nanmean(np.abs(vort_peak_history[p]))):.2f}")
    for k, lbl in enumerate(R_labels):
        print(f"    {lbl}: mean={R_stats[p, k, 0]:.4f},  "
              f"std={R_stats[p, k, 1]:.4f}")
print(f"  vort_rms_tavg = {vort_rms_tavg:.4f}")
print(f"\nAntipodality:")
for k, lbl in enumerate(antipodal_stat_labels):
    print(f"  {lbl}: mean={antipodal_stats[k, 0]:+.4f},  "
          f"std={antipodal_stats[k, 1]:.4f}")

### --- outputs (single combined file) --- ###
processed = {
    # per-frame tracking
    'theta_locs'        : theta_locs,
    'phi_locs'          : phi_locs,
    'vort_peak_history' : vort_peak_history,
    'sign_history'      : sign_history,
    'tw'                : tw,
    'ws'                : ws,
    # length scales
    'R_history'         : R_history,
    'R_stats'           : R_stats,
    'R_labels'          : R_labels,
    'R_stat_labels'     : ['mean', 'std'],
    'R_threshold_fractions': peak_thresholds,
    # CPC-frame profile
    'vort_radial_tavg'  : vort_radial_tavg,
    'rho_fit_grid'      : rho_fit_grid,
    'alpha_fit_grid'    : alpha_fit_grid,
    'n_valid_for_avg'   : n_valid_for_avg,
    # whole sphere
    'vort_rms_history'  : vort_rms_history,
    'vort_rms_tavg'     : vort_rms_tavg,
    # position histograms (combined into main file)
    'position_hist'     : position_hist,
    'phi_bin_edges'     : phi_bin_edges,
    'theta_bin_edges'   : theta_bin_edges,
    'phi_bin_centers'   : phi_bin_centers,
    'theta_bin_centers' : theta_bin_centers,
    'n_frames_hist'     : n_frames_hist,
    # antipodality
    'antipodal_gc_dist'      : antipodal_gc_dist,
    'antipodal_delta_theta'  : antipodal_delta_theta,
    'antipodal_delta_phi'    : antipodal_delta_phi,
    'relative_zonal_phase'   : relative_zonal_phase,
    'antipodal_stats'        : antipodal_stats,
    'antipodal_stat_labels'  : antipodal_stat_labels,
    # grids and metadata
    'phi_1d': phi_1d, 'theta_1d': theta_1d,
    'gamma': gamma, 'alpha': alpha, 'eps': eps,
    'cap_colat': cap_colat,
    'Nphi': Nphi_grid, 'Ntheta': Ntheta_grid,
    'vort_task': vort_task,
    'output_suffix': output_suffix,
}
print(f"\nSaving: {output_path}")
np.save(output_path, processed)

### --- diagnostic figure --- ###
if plot_path is not None:
    print(f"Saving plot: {plot_path}")
    fig = plt.figure(figsize=(15, 13), constrained_layout=True)
    gs  = fig.add_gridspec(3, 3)
    pole_names  = ['N', 'S']
    pole_colors = ['C0', 'C3']
    R_colors    = ['C0', 'C1', 'C3', 'C5']

    # (0,0): theta(t) for both poles
    ax = fig.add_subplot(gs[0, 0])
    for p in range(n_poles):
        ax.plot(tw, theta_locs[p], color=pole_colors[p], lw=0.6,
                label=fr'$\theta_{pole_names[p]}$')
    ax.axhline(cap_colat, color='gray', ls=':', lw=0.7)
    ax.axhline(np.pi - cap_colat, color='gray', ls=':', lw=0.7)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\theta$ (rad)')
    ax.set_title('CPC colatitude trajectories')
    ax.set_ylim([0, np.pi]); ax.invert_yaxis()
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) and (0,2): polar projection per pole, with 2D residency
    # histogram in the background and the per-frame trajectory scatter
    # on top (time-colored).
    for p, gs_col in [(0, 1), (1, 2)]:
        ax = fig.add_subplot(gs[0, gs_col], projection='polar')
        # for the S pole, use (phi, pi - theta) so radial axis grows away
        # from that pole; flip the theta bin edges to match
        if p == 0:
            theta_edges_plot = theta_bin_edges
            radial_scatter   = theta_locs[p]
            radial_lbl       = r'$\theta$'
        else:
            theta_edges_plot = np.pi - theta_bin_edges
            radial_scatter   = np.pi - theta_locs[p]
            radial_lbl       = r'$\pi - \theta$'
        phi_mesh, th_mesh = np.meshgrid(phi_bin_edges, theta_edges_plot,
                                          indexing='ij')
        h = position_hist[p]
        ax.pcolormesh(phi_mesh, th_mesh, np.log10(h + 1),
                       shading='flat', cmap='magma', alpha=0.55,
                       vmin=0, vmax=np.log10(max(h.max(), 1) + 1))
        ax.scatter(phi_locs[p], radial_scatter, s=3, c=tw, cmap='viridis',
                   alpha=0.6)
        ax.set_ylim([0, cap_colat * 1.2])
        ax.set_title(fr'{pole_names[p]}:  ($\phi$, {radial_lbl})  '
                     fr'+ 2D residency', fontsize=10)

    # (1,0) and (1,1): R(t) per pole
    for p, gs_col in [(0, 0), (1, 1)]:
        ax = fig.add_subplot(gs[1, gs_col])
        for k, lbl in enumerate(R_labels):
            ax.plot(tw, R_history[p, k], color=R_colors[k], lw=0.6,
                    label=fr'${lbl}$')
        ax.set_xlabel('t'); ax.set_ylabel(r'$R$ (rad)')
        ax.set_title(f'{pole_names[p]}: $R(t)$')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # (1,2): time-averaged radial profiles
    ax = fig.add_subplot(gs[1, 2])
    u_rms_scale = float(np.sqrt(eps / alpha))
    for p in range(n_poles):
        ax.plot(rho_fit_grid[1:], vort_radial_tavg[p, 1:] / u_rms_scale,
                color=pole_colors[p], lw=1.5,
                label=fr'{pole_names[p]} (signed)')
        ax.axvline(R_stats[p, 0, 0], color=pole_colors[p], ls='--', lw=0.7)
        ax.axvline(R_stats[p, 2, 0], color=pole_colors[p], ls=':',  lw=0.7)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\rho_\mathrm{gc}$ (rad)')
    ax.set_ylabel(r'$\bar\omega / \sqrt{\epsilon/\alpha}$')
    ax.set_title(r'Time-avg radial profiles')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    # (2,0): antipodal great-circle distance
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(tw, antipodal_gc_dist, color='C2', lw=0.6)
    ax.axhline(antipodal_stats[0, 0], color='C2', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {antipodal_stats[0, 0]:.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel('great-circle dist (rad)')
    ax.set_title('Antipodal distance')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2,1): antipodal colatitude deviation
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(tw, antipodal_delta_theta, color='C0', lw=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(antipodal_stats[1, 0], color='C0', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {antipodal_stats[1, 0]:+.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\theta_N + \theta_S - \pi$ (rad)')
    ax.set_title(r'Antipodal $\Delta\theta$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2,2): antipodal Delta_phi + raw zonal phase difference
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(tw, antipodal_delta_phi,  color='C3', lw=0.6,
            label=r'$\Delta\phi$ (antipodal)')
    ax.plot(tw, relative_zonal_phase, color='C4', lw=0.6, alpha=0.6,
            label=r'wrap($\phi_N - \phi_S$)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('t'); ax.set_ylabel('rad')
    ax.set_title(r'Longitudinal alignment')
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Spherical locator — {output_suffix}', fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

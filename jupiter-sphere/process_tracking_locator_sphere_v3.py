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

--- Usage ---
Usage:
    process_tracking_locator_sphere_v1.py <hdf5_file> [options]

The <hdf5_file> argument accepts either a single HDF5 file or a glob
pattern matching multiple set files (e.g. 'snapshots/snapshots_s*.h5');
matched files are sorted by their trailing integer index and their
writes are processed consecutively as if they belonged to one file.
No external merging step is required.

Arguments:
    <hdf5_file>   single HDF5 file or glob pattern matching set files

Options:
    --output=<str>           output .npy path; 'auto' uses output_suffix [default: auto]
    --plot=<str>             POSITION figure path; 'auto'; 'none' to skip
                             [default: auto]
    --plot_amp=<str>         AMPLITUDE/PROFILE figure path; 'auto'; 'none' to skip
                             [default: auto]
    --output_prefix=<str>    prefix used when paths are 'auto' [default: processed_tracking_locator_sphere]
    --output_suffix=<str>    suffix; 'auto' derives from filename [default: auto]

    --t_start=<float>        sim time to begin tracking [default: 0.]
    --t_end=<float>          sim time to stop tracking  [default: 1e9]
    --tavg_window=<str>      window over which to compute time-averages
                             (R_stats, vort_radial_tavg, vort_rms_tavg,
                             antipodal_stats, mean lines on plots).
                             Accepts:
                                'data'           - full tracking range
                                'last_half'      - second half of range
                                'last_<float>'   - last <float> sim-time units
                                '<start>:<end>'  - explicit sim-time bounds
                             [default: last_half]

    --vort_task=<str>        HDF5 task name for vorticity [default: vorticity]

    --gamma=<str>            gamma (= 2*Omega); 'auto' uses 2*1186 [default: auto]
    --alpha=<float>          friction [default: 0.0333333]
    --eps=<float>            energy injection rate [default: 1.0]

    --cap_colat=<str>        outer colatitude of each polar cap (rad), clamped
                             to <= pi/2; 'auto' uses max(2 * (spherical Rhines
                             angle), min_cap_colat) for gamma>0, or pi/2
                             (hemispheric) for gamma=0 [default: auto]
    --min_cap_colat=<float>  minimum cap_colat (rad) used when 2*L_rh would
                             be smaller; rotating cases only.  Default
                             0.2618 ~ pi/12 ~ 15 deg.  Overridden by the
                             hemispheric setting when gamma <= 0 [default: 0.2618]

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

from tracking_locator_sphere_helpers_v2 import (
    open_snapshot_set,
    great_circle_distance,
    find_threshold_crossing,
    cpc_frame_to_lab,
    find_max_abs_in_cap,
    refine_extremum_via_spline,
)

### parse args ###
args = docopt(__doc__)
print(args)

file_str       = args['<hdf5_file>']
output_arg     = args['--output']
plot_arg       = args['--plot']
plot_amp_arg   = args['--plot_amp']
output_prefix  = args['--output_prefix']
output_suffix_arg = args['--output_suffix']
t_start        = float(args['--t_start'])
t_end          = float(args['--t_end'])
tavg_window_arg = args['--tavg_window']
vort_task      = args['--vort_task']
gamma_arg      = args['--gamma']
alpha          = float(args['--alpha'])
eps            = float(args['--eps'])
cap_colat_arg  = args['--cap_colat']
min_cap_colat  = float(args['--min_cap_colat'])
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
    path_or_default(plot_arg, f"{output_prefix}_position_{output_suffix}.png")
plot_amp_path = None if plot_amp_arg.lower() in ('none', '') else \
    path_or_default(plot_amp_arg, f"{output_prefix}_amplitude_{output_suffix}.png")
print(f"output .npy:    {output_path}")
print(f"position plot:  {plot_path}")
print(f"amplitude plot: {plot_amp_path}")

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
        # cap is 2*L_rh, floored by min_cap_colat (so the search isn't
        # too tight when the CPC's natural confinement is very strong),
        # and ceilinged by pi/2 (hemispheric upper bound).
        cap_colat = float(min(max(2 * L_rh, min_cap_colat), np.pi / 2))
        print(f"cap_colat (auto): U_rms={U_rms:.4f}, "
              f"L_rh_flat={L_rh_flat:.4f}, L_rh_sphere={L_rh:.4f}, "
              f"min_cap_colat={min_cap_colat:.4f}, "
              f"cap_colat={cap_colat:.4f}")
else:
    cap_colat = float(cap_colat_arg)
    if cap_colat > np.pi / 2:
        cap_colat = np.pi / 2
        print(f"  warning: cap_colat capped at pi/2 (hemispheric upper bound)")
    print(f"cap_colat (CLI): {cap_colat:.4f}")

### open HDF5 file(s) ###
snap = open_snapshot_set(file_str, vort_task)
files, vort_dsets       = snap.files, snap.vort_dsets
t_all                   = snap.t_all
file_idxs, local_idxs   = snap.file_idxs, snap.local_idxs
Nphi_grid, Ntheta_grid  = snap.Nphi_grid, snap.Ntheta_grid
print(f"HDF5: {len(files)} file(s), {len(t_all)} total writes, "
      f"t in [{t_all[0]:.3f}, {t_all[-1]:.3f}], "
      f"grid ({Nphi_grid} phi, {Ntheta_grid} theta)")

### Dedalus basis (matching Aishani's IVP) ###
# Used here both to provide the lab-frame (phi, theta) grids and to do
# the per-frame sphere-averaged rms via d3.Average.
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

### resolve time-averaging window ###
# tavg_window picks the sub-range of tw over which time-mean statistics
# (R_stats, vort_radial_tavg, vort_rms_tavg, antipodal_stats, mean lines
# on plots) are computed. The per-frame arrays themselves still span the
# full tw range so that the time series plots are uninterrupted.
def resolve_tavg_window(arg, tw):
    arg_lo = arg.lower().strip()
    t0_data, t1_data = float(tw[0]), float(tw[-1])
    if arg_lo == 'data':
        return t0_data, t1_data, 'data'
    if arg_lo == 'last_half':
        return 0.5 * (t0_data + t1_data), t1_data, 'last_half'
    if arg_lo.startswith('last_'):
        try:
            dur = float(arg_lo[len('last_'):])
        except ValueError:
            raise ValueError(f"Couldn't parse --tavg_window='{arg}'")
        return max(t0_data, t1_data - dur), t1_data, f'last_{dur:g}'
    if ':' in arg_lo:
        parts = arg_lo.split(':')
        if len(parts) != 2:
            raise ValueError(f"Couldn't parse --tavg_window='{arg}'")
        try:
            t0 = float(parts[0]) if parts[0] != '' else t0_data
            t1 = float(parts[1]) if parts[1] != '' else t1_data
        except ValueError:
            raise ValueError(f"Couldn't parse --tavg_window='{arg}'")
        return t0, t1, f'{t0:g}:{t1:g}'
    raise ValueError(f"Unknown --tavg_window='{arg}'")

tavg_t0, tavg_t1, tavg_window_src = resolve_tavg_window(tavg_window_arg, tw)
tavg_mask = (tw >= tavg_t0 - 1e-12) & (tw <= tavg_t1 + 1e-12)
n_tavg    = int(tavg_mask.sum())
print(f"tavg_window: '{tavg_window_src}' -> t in "
      f"[{tavg_t0:.3f}, {tavg_t1:.3f}] ({n_tavg}/{nw} frames)")
if n_tavg == 0:
    raise ValueError("tavg_window selected zero frames; widen it.")

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

### main loop ###
prog_cad = max(1, nw // 50)
print(f"\nStarting per-frame locator (nw={nw}) ...")
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")

    # look up which file this global frame index belongs to, then load;
    # raw vort is in Dedalus order (descending theta), so we set the
    # Dedalus field for the sphere average and then flip to ascending
    # for all subsequent local processing.
    vort_raw = np.array(vort_dsets[file_idxs[w]][local_idxs[w]])
    if vort_raw.shape == (Nphi_grid, Ntheta_grid):
        vort_field.change_scales(1)
    elif vort_raw.shape == (int(dealias*Nphi_grid), int(dealias*Ntheta_grid)):
        vort_field.change_scales(dealias)
    else:
        print("should never happen")
        print(vort_raw.shape, (int(dealias*Nphi_grid), int(dealias*Ntheta_grid)))
        raise
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
        # grid-point extremum (coarse), then sub-grid refinement via local
        # spline + L-BFGS-B for accurate (theta, phi, vort)
        phi_idx, theta_idx, _, sign = find_max_abs_in_cap(
            vort_g, cap_idxs_per_pole[p])
        theta_p, phi_p, vort_val = refine_extremum_via_spline(
            theta_1d, phi_1d, vort_g, phi_idx, theta_idx)
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
            # only accumulate the time-avg radial profile over the
            # selected tavg_window
            if tavg_mask[i] and np.all(np.isfinite(vort_radial)):
                vort_radial_sum[p] += vort_radial
                n_valid_for_avg[p] += 1

        # 2D position histogram (only frames inside tavg_window so the
        # residency map matches the rest of the time-mean statistics)
        if tavg_mask[i]:
            phi_bin   = max(0, min(
                int(np.digitize(phi_p % (2 * np.pi), phi_bin_edges)) - 1,
                n_phi_bins - 1))
            theta_bin = max(0, min(
                int(np.digitize(theta_p, theta_bin_edges)) - 1,
                n_theta_bins - 1))
            position_hist[p, phi_bin, theta_bin] += 1
            n_frames_hist[p] += 1

for fh in files:
    fh.close()
print("Per-frame loop done.")

### post-loop computations ###
# All time-mean statistics below restrict to frames inside tavg_mask
# (the --tavg_window slice).
vort_radial_tavg = vort_radial_sum / np.maximum(n_valid_for_avg[:, None], 1)
vort_rms_tavg    = float(np.nanmean(vort_rms_history[tavg_mask]))

# R_stats[pole, R_type, stat]: stat 0 = mean, 1 = std (over tavg window)
R_stats = np.full((n_poles, n_R_types, 2), np.nan)
for p in range(n_poles):
    for k in range(n_R_types):
        vals = R_history[p, k, tavg_mask]
        fin  = vals[np.isfinite(vals)]
        if len(fin) > 0:
            R_stats[p, k, 0] = float(np.mean(fin))
            R_stats[p, k, 1] = float(np.std(fin))

# vortex amplitude stats (over tavg window)
vort_peak_stats = np.full((n_poles, 2), np.nan)
for p in range(n_poles):
    vals = vort_peak_history[p, tavg_mask]
    fin  = vals[np.isfinite(vals)]
    if len(fin) > 0:
        vort_peak_stats[p, 0] = float(np.mean(np.abs(fin)))
        vort_peak_stats[p, 1] = float(np.std(np.abs(fin)))

# antipodality (per-frame for full tw; stats over tavg window)
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

def _mean_std_over_mask(arr, mask):
    sub = arr[mask]
    fin = sub[np.isfinite(sub)]
    if len(fin) == 0:
        return np.nan, np.nan
    return float(np.mean(fin)), float(np.std(fin))

antipodal_stats = np.array([
    list(_mean_std_over_mask(antipodal_gc_dist,     tavg_mask)),
    list(_mean_std_over_mask(antipodal_delta_theta, tavg_mask)),
    list(_mean_std_over_mask(antipodal_delta_phi,   tavg_mask)),
    list(_mean_std_over_mask(relative_zonal_phase,  tavg_mask)),
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

### save (single combined file) ###
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
    # vortex amplitude summary
    'vort_peak_stats'   : vort_peak_stats,  # (n_poles, 2): <|peak|>, std
    # tavg window metadata
    'tavg_t0'           : tavg_t0,
    'tavg_t1'           : tavg_t1,
    'tavg_window_src'   : tavg_window_src,
    'tavg_mask'         : tavg_mask,
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

### diagnostic plots (two figures: position+antipodality, amplitude+profile) ###
pole_names  = ['N', 'S']
pole_colors = ['C0', 'C3']
R_colors    = ['C0', 'C1', 'C3', 'C5']

# Folded S colatitude (distance-from-own-pole) used in the overlapping
# trajectory and histogram plots.
theta_S_folded = np.pi - theta_locs[1, :]
theta_folded   = [theta_locs[0, :], theta_S_folded]
fold_labels    = [r'$\theta_N$', r'$\pi - \theta_S$']
# Auto y-limit: take max excursion over the tavg window, then pad.
def _theta_fold_ylim():
    finite = []
    for arr in theta_folded:
        sub = arr[tavg_mask]
        finite.append(sub[np.isfinite(sub)])
    if any(len(x) > 0 for x in finite):
        ymax = max(np.nanmax(x) for x in finite if len(x) > 0)
    else:
        ymax = cap_colat
    return [0, max(min(np.pi / 2, ymax * 1.15), 0.05)]
theta_fold_ylim = _theta_fold_ylim()

def _shade_tavg(ax):
    """Shade the tavg window on a t-vs-y plot for orientation."""
    ax.axvspan(tavg_t0, tavg_t1, color='gray', alpha=0.08, lw=0,
                zorder=0)

### =========================== POSITION FIGURE =========================== ###
if plot_path is not None:
    print(f"Saving position plot: {plot_path}")
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs  = fig.add_gridspec(3, 3)

    # (0, 0) — folded θ trajectories, both poles overlapping near zero
    ax = fig.add_subplot(gs[0, 0])
    for p in range(n_poles):
        ax.plot(tw, theta_folded[p], color=pole_colors[p], lw=0.6,
                label=fold_labels[p])
    ax.axhline(cap_colat, color='gray', ls=':', lw=0.7,
                label=fr'cap_colat = {cap_colat:.3f}')
    _shade_tavg(ax)
    ax.set_xlabel('t'); ax.set_ylabel('colat from own pole (rad)')
    ax.set_title(r'CPC distance-from-pole trajectories (folded)')
    ax.set_ylim(theta_fold_ylim)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0, 1) / (0, 2) — polar scatter per pole, colored by time; no
    # residency pcolormesh underneath (the 1D histograms below give
    # cleaner marginals than the gray map).
    for p, gs_col in [(0, 1), (1, 2)]:
        ax = fig.add_subplot(gs[0, gs_col], projection='polar')
        radial = theta_folded[p]
        sc = ax.scatter(phi_locs[p], radial, s=4, c=tw, cmap='viridis',
                         alpha=0.7)
        ax.set_ylim([0, theta_fold_ylim[1]])
        ax.set_title(fr'{pole_names[p]}: $(\varphi, $'
                      + (r'$\theta_N$' if p == 0 else r'$\pi-\theta_S$')
                      + r'$)$  trajectory', fontsize=10)
        cb = fig.colorbar(sc, ax=ax, pad=0.08, fraction=0.04)
        cb.set_label('sim time t', fontsize=8)
        # Note for the reader: time increases dark-blue -> yellow.

    # (1, 0) — 1D θ histogram (folded) for both poles overlaid
    ax = fig.add_subplot(gs[1, 0])
    th_bins = np.linspace(0, theta_fold_ylim[1], 50)
    for p in range(n_poles):
        sub = theta_folded[p][tavg_mask]
        sub = sub[np.isfinite(sub)]
        ax.hist(sub, bins=th_bins, alpha=0.55, color=pole_colors[p],
                label=fr'{pole_names[p]}: {fold_labels[p]}', density=True)
    ax.set_xlabel('colat from own pole (rad)'); ax.set_ylabel('density')
    ax.set_title(r'1D colatitude histogram (tavg window)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1, 1) / (1, 2) — 1D φ histogram per pole
    for p, gs_col in [(0, 1), (1, 2)]:
        ax = fig.add_subplot(gs[1, gs_col])
        sub = phi_locs[p][tavg_mask]
        sub = sub[np.isfinite(sub)]
        ax.hist(sub, bins=np.linspace(0, 2*np.pi, 40), color=pole_colors[p],
                alpha=0.8, density=True)
        ax.set_xlabel(r'$\varphi$ (rad)'); ax.set_ylabel('density')
        ax.set_xlim([0, 2 * np.pi])
        ax.set_xticks(np.arange(0, 2*np.pi + 0.01, np.pi / 2))
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$',
                              r'$2\pi$'])
        ax.set_title(fr'{pole_names[p]}: 1D $\varphi$ histogram '
                      f'(tavg window)')
        ax.grid(True, alpha=0.3)

    # (2, 0) — antipodal great-circle distance.  This is the angular
    # distance between CPC_N's position and the antipode of CPC_S; zero
    # if N and S are perfectly antipodal regardless of which axis they
    # share.  Bounded above by pi (~3.14).
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(tw, antipodal_gc_dist, color='C2', lw=0.6)
    ax.axhline(antipodal_stats[0, 0], color='C2', ls='--', lw=1.0,
                label=fr'$\langle\rangle = {antipodal_stats[0, 0]:.3f}$')
    _shade_tavg(ax)
    ax.set_xlabel('t')
    ax.set_ylabel('great-circle dist (rad)')
    ax.set_title(r'Antipodal distance:  $\angle($CPC$_N$, antipode of '
                  r'CPC$_S)$,  0 $\Leftrightarrow$ antipodal')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2, 1) — antipodal colatitude deviation.  This is θ_N + θ_S - π;
    # zero if each CPC sits the same angular distance from its own pole.
    # Decouples the colatitude question from the longitudinal one.
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(tw, antipodal_delta_theta, color='C0', lw=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(antipodal_stats[1, 0], color='C0', ls='--', lw=1.0,
                label=fr'$\langle\rangle = {antipodal_stats[1, 0]:+.3f}$')
    _shade_tavg(ax)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\theta_N + \theta_S - \pi$ (rad)')
    ax.set_title(r'Colatitude mismatch:  $\theta_N + \theta_S - \pi$,  '
                  r'0 $\Leftrightarrow$ equal distance from own pole')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (2, 2) — wrap(φ_N − φ_S): just the angular separation in longitude.
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(tw, relative_zonal_phase, color='C4', lw=0.6)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axhline(antipodal_stats[3, 0], color='C4', ls='--', lw=1.0,
                label=fr'$\langle\rangle = {antipodal_stats[3, 0]:+.3f}$')
    _shade_tavg(ax)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\mathrm{wrap}(\varphi_N - \varphi_S)$ (rad)')
    ax.set_title(r'Longitudinal separation:  '
                  r'$\mathrm{wrap}(\varphi_N - \varphi_S)$')
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Sphere locator — position & antipodality — '
                  f'{output_suffix}\n'
                  f'tavg window: {tavg_window_src} '
                  f'(t in [{tavg_t0:.2f}, {tavg_t1:.2f}], '
                  f'{n_tavg}/{nw} frames)',
                  fontsize=10)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

### ======================== AMPLITUDE / PROFILE FIGURE ======================== ###
if plot_amp_path is not None:
    print(f"Saving amplitude plot: {plot_amp_path}")
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3)
    u_rms_scale = float(np.sqrt(eps / alpha))

    # (0, 0) — signed peak vorticity time series
    ax = fig.add_subplot(gs[0, 0])
    for p in range(n_poles):
        ax.plot(tw, vort_peak_history[p] / u_rms_scale,
                color=pole_colors[p], lw=0.6,
                label=fr'{pole_names[p]}:  '
                       fr'$\langle|\omega_0|\rangle/\sqrt{{\epsilon/\alpha}}'
                       fr'={vort_peak_stats[p,0]/u_rms_scale:.2f}$')
    ax.axhline(0, color='gray', lw=0.5)
    _shade_tavg(ax)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\omega_0(t)\,/\,\sqrt{\epsilon/\alpha}$')
    ax.set_title('Signed CPC peak vorticity (normalized)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0, 1) — 1D histogram of |peak| (over tavg window)
    ax = fig.add_subplot(gs[0, 1])
    peak_bins_data = np.concatenate([
        np.abs(vort_peak_history[p][tavg_mask]) for p in range(n_poles)])
    peak_bins_data = peak_bins_data[np.isfinite(peak_bins_data)]
    if len(peak_bins_data) > 0:
        bins = np.linspace(0, np.nanmax(peak_bins_data) * 1.05, 40)
    else:
        bins = 40
    for p in range(n_poles):
        sub = np.abs(vort_peak_history[p][tavg_mask])
        sub = sub[np.isfinite(sub)]
        ax.hist(sub / u_rms_scale, bins=bins / u_rms_scale,
                color=pole_colors[p], alpha=0.55, label=pole_names[p],
                density=True)
    ax.set_xlabel(r'$|\omega_0|\,/\,\sqrt{\epsilon/\alpha}$')
    ax.set_ylabel('density')
    ax.set_title('Histogram of $|\\omega_0|$ (tavg window)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0, 2) — time-avg radial profile (signed; both poles)
    ax = fig.add_subplot(gs[0, 2])
    for p in range(n_poles):
        ax.plot(rho_fit_grid[1:], vort_radial_tavg[p, 1:] / u_rms_scale,
                color=pole_colors[p], lw=1.5,
                label=fr'{pole_names[p]} (signed)')
        ax.axvline(R_stats[p, 0, 0], color=pole_colors[p], ls='--', lw=0.7)
        ax.axvline(R_stats[p, 2, 0], color=pole_colors[p], ls=':',  lw=0.7)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\rho_\mathrm{gc}$ (rad)')
    ax.set_ylabel(r'$\bar\omega\,/\,\sqrt{\epsilon/\alpha}$')
    ax.set_title('Time-avg radial profile (signed)')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    # (1, 0) / (1, 1) — R(t) per pole, all four thresholds
    for p, gs_col in [(0, 0), (1, 1)]:
        ax = fig.add_subplot(gs[1, gs_col])
        for k, lbl in enumerate(R_labels):
            ax.plot(tw, R_history[p, k], color=R_colors[k], lw=0.6,
                    label=fr'${lbl}$')
            ax.axhline(R_stats[p, k, 0], color=R_colors[k], ls='--',
                        lw=0.8, alpha=0.7)
        _shade_tavg(ax)
        ax.set_xlabel('t'); ax.set_ylabel(r'$R$ (rad)')
        ax.set_title(fr'{pole_names[p]}:  $R(t)$ at each threshold')
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # (1, 2) — histogram of R values (one panel; pole + threshold encoded)
    ax = fig.add_subplot(gs[1, 2])
    R_max_for_bins = 0.0
    for p in range(n_poles):
        for k in range(n_R_types):
            sub = R_history[p, k][tavg_mask]
            sub = sub[np.isfinite(sub)]
            if len(sub) > 0:
                R_max_for_bins = max(R_max_for_bins, float(np.nanmax(sub)))
    R_bins = np.linspace(0, max(R_max_for_bins * 1.05, 1e-3), 40)
    # Iterate threshold-major, pole-minor so the legend groups by R type.
    for k, lbl in enumerate(R_labels):
        for p in range(n_poles):
            sub = R_history[p, k][tavg_mask]
            sub = sub[np.isfinite(sub)]
            if len(sub) == 0:
                continue
            ls = '-' if p == 0 else '--'
            ax.hist(sub, bins=R_bins, histtype='step', density=True,
                    color=R_colors[k], lw=1.1, ls=ls,
                    label=fr'{lbl} ({pole_names[p]})')
    ax.set_xlabel(r'$R$ (rad)'); ax.set_ylabel('density')
    ax.set_title('Histogram of $R$ values (tavg window)')
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Sphere locator — amplitude & length scales — '
                  f'{output_suffix}\n'
                  f'tavg window: {tavg_window_src} '
                  f'(t in [{tavg_t0:.2f}, {tavg_t1:.2f}], '
                  f'{n_tavg}/{nw} frames)',
                  fontsize=10)
    fig.savefig(plot_amp_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_amp_path}")

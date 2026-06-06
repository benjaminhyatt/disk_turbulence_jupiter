"""
RW amplitude via PER-FRAME parametric CPC template + hybrid LSQ (v4).

Same hybrid LSQ structure as v3 (psi-space biorthogonal A_n, omega-space
cost function for a), but the template ω_T(t) is now built per frame
from a parametric fit to the snapshot's alpha-averaged CPC profile,
rather than the time-averaged shape from the locator.

Per non-glitch frame:
  1.  Load omega(t) from HDF5.
  2.  Compute the snapshot's alpha-averaged radial vorticity profile
      omega_bar(rho, t) by interpolating to a CPC-frame grid.
  3.  Fit a parametric profile (default: Lamb-Oseen, two parameters
      omega_0, L) to omega_bar over rho in [0, rho_fit_max(t)], where
      rho_fit_max(t) is by default the per-frame R_urms(t) from the
      locator.
  4.  Build omega_T(t) on the lab-frame Dedalus grid from the fitted
      analytic profile.
  5.  Solve Poisson for psi(t) and psi_T(t).
  6.  Run the joint LSQ exactly as in v3:
        a(t) = <omega_T_eff, omega_eff> / ||omega_T_eff||^2
        A_n(t) = (b_n - a*D_n)/biortho_n

The CPC-localized SNR metric is:
    SNR_a^local = |a| * ||omega_T_eff||_loc / ||omega_residual||_loc
where ||.||_loc is L2 over a CPC-frame disk of radius snr_local_radius
(default 0.4, the locator's rho_max_template).  This excludes far-field
turbulence from the noise estimate.

Usage:
    process_rw_amplitude_subtracted_v4.py <hdf5_file> <evp_file> <tracking_file> [options]

Arguments as in v3; new options described below.

Options:
    --cpc_model=<str>            parametric model for the CPC profile;
                                 one of {lamb_oseen, rankine_smooth,
                                 rankine_strict, empirical}.
                                 'empirical' falls back to v3 behavior
                                 (uses vort_radial_tavg directly).
                                 [default: lamb_oseen]
    --cpc_fit_rho_max=<str>      upper rho bound for the per-frame fit;
                                 one of R_urms, R_tenth, R_fifth, R_half
                                 (per-frame history), corresponding
                                 R_*_mean (constant), or a literal float
                                 [default: R_urms]
    --snr_local_radius=<str>     radius (CPC-frame) for the local SNR
                                 norms; same parsing as above
                                 [default: 0.4]

    --n_candidate_modes=<int>    [default: 4]
    --candidate_mode_idxs=<str>  [default: auto]
    --fft_file=<str>             [default: none]
    --fft_mode_idx=<int>         [default: 0]
    --match_tol=<float>          [default: 1e-6]
    --t_start=<str>              [default: tracking]
    --t_end=<str>                [default: tracking]
    --output=<str>               [default: auto]
    --plot=<str>                 [default: auto]
    --output_prefix=<str>        [default: processed_rw_amplitude_subtracted_v4]
    --Nphi=<str>                 [default: auto]
    --Nr=<str>                   [default: auto]
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read args ###
args = docopt(__doc__)
print(args)

hdf5_file        = args['<hdf5_file>']
evp_file         = args['<evp_file>']
tracking_file    = args['<tracking_file>']

cpc_model        = args['--cpc_model'].lower()
cpc_fit_rho_max_arg = args['--cpc_fit_rho_max']
snr_local_radius_arg = args['--snr_local_radius']

n_candidate_modes = int(args['--n_candidate_modes'])
cand_idxs_arg     = args['--candidate_mode_idxs']

fft_file_arg     = args['--fft_file']
fft_mode_idx     = int(args['--fft_mode_idx'])
match_tol        = float(args['--match_tol'])

t_start_arg      = args['--t_start']
t_end_arg        = args['--t_end']

output_arg       = args['--output']
plot_arg         = args['--plot']
output_prefix    = args['--output_prefix']
Nphi_arg         = args['--Nphi']
Nr_arg           = args['--Nr']

dealias          = 3/2

valid_models = ('lamb_oseen', 'rankine_smooth', 'rankine_strict', 'empirical')
if cpc_model not in valid_models:
    raise ValueError(f"--cpc_model must be one of {valid_models}; got "
                     f"'{cpc_model}'")

def extract_output_suffix(file_path):
    basename = file_path.split('/')[-1]
    if basename.endswith('.npy'):
        basename = basename[:-4]
    for prefix in ('processed_tracking_locator_',
                   'processed_tracking_',
                   'processed_rossby_projection_fft_',
                   'processed_rossby_projection_',
                   'processed_rossby_evp_',
                   'analysis_'):
        if prefix in basename:
            basename = basename.split(prefix, 1)[1]
            break
    return basename

output_suffix = extract_output_suffix(tracking_file)
print(f"output_suffix: {output_suffix}")

def auto_or(arg, default_template):
    return default_template if arg.lower() == 'auto' else arg

output_path = auto_or(output_arg, f"{output_prefix}_{output_suffix}.npy")
if plot_arg.lower() in ('none', ''):
    plot_path = None
else:
    plot_path = auto_or(plot_arg, f"{output_prefix}_{output_suffix}.png")
print(f"output .npy: {output_path}")
print(f"plot path:   {plot_path}")
print(f"cpc_model:   {cpc_model}")

### load tracking ###
logger.info(f"Loading tracking file: {tracking_file}")
tracking      = np.load(tracking_file, allow_pickle=True)[()]
r_locs        = np.array(tracking['r_locs'],   dtype=float)
phi_locs      = np.array(tracking['phi_locs'], dtype=float)
tw_track      = np.array(tracking['tw'],       dtype=float)
ws_track      = np.array(tracking['ws'],       dtype=int)
glitch_flags  = np.array(tracking['glitch_flags'], dtype=bool)
n_total_track = len(r_locs)
n_glitch      = int(np.sum(glitch_flags))
r_CPC_mean    = float(np.mean(r_locs[~glitch_flags]))

for key in ('rho_fit_grid', 'vort_radial_tavg', 'omega_peak_history'):
    if key not in tracking:
        raise KeyError(f"Tracking file missing '{key}'.")
rho_fit_grid       = np.array(tracking['rho_fit_grid'],       dtype=float)
vort_radial_tavg   = np.array(tracking['vort_radial_tavg'],   dtype=float)
omega_peak_history = np.array(tracking['omega_peak_history'], dtype=float)
peak0              = float(vort_radial_tavg[0])
rho_max_template   = float(rho_fit_grid[-1])
mean_peak          = float(np.nanmean(omega_peak_history))
print(f"tracking: {n_total_track} frames, {n_glitch} glitches, "
      f"<r_CPC>={r_CPC_mean:.4f}")
print(f"locator: peak0={peak0:.4f}, rho_max_template={rho_max_template:.3f}")

### time window ###
if t_start_arg.lower() == 'tracking':
    t_start_eff = float(tw_track[0])
else:
    t_start_eff = float(t_start_arg)
if t_end_arg.lower() == 'tracking':
    t_end_eff = float(tw_track[-1])
else:
    t_end_eff = float(t_end_arg)
print(f"time window: [{t_start_eff:.3f}, {t_end_eff:.3f}]")

in_window  = (tw_track >= t_start_eff - 1e-9) & (tw_track <= t_end_eff + 1e-9)
sub_idxs   = np.where(in_window)[0]
ws_sub     = ws_track[sub_idxs]
tw_sub     = tw_track[sub_idxs]
r_sub      = r_locs[sub_idxs]
phi_sub    = phi_locs[sub_idxs]
glitch_sub = glitch_flags[sub_idxs]
peak_sub   = omega_peak_history[sub_idxs]
nw         = len(ws_sub)
print(f"processing nw={nw} frames")

### resolve R-scale arrays for fit range and SNR radius ###
def resolve_R_array(arg, tracking, sub_idxs, nw, default_fallback):
    """Return a per-frame array of length nw based on arg parsing."""
    arg_lo = arg.lower()
    history_map = {'r_urms': ('R_urms_history',  'R_urms_mean'),
                   'r_half':  ('R_half_history',  'R_half_mean'),
                   'r_fifth': ('R_fifth_history', 'R_fifth_mean'),
                   'r_tenth': ('R_tenth_history', 'R_tenth_mean')}
    mean_set = ('r_urms_mean', 'r_half_mean', 'r_fifth_mean', 'r_tenth_mean')
    if arg_lo in history_map:
        hist_key, mean_key = history_map[arg_lo]
        if hist_key in tracking:
            arr = np.array(tracking[hist_key])[sub_idxs].astype(float)
            mean_val = float(tracking[mean_key]) if mean_key in tracking \
                       else default_fallback
            arr = np.where(np.isfinite(arr) & (arr > 0), arr, mean_val)
            return arr, 'per_frame_' + arg_lo
        elif mean_key in tracking:
            return np.full(nw, float(tracking[mean_key])), 'mean_' + arg_lo
        else:
            return np.full(nw, default_fallback), 'fallback'
    elif arg_lo in mean_set:
        # uppercase the key
        key = arg.upper() if arg.startswith('R_') else 'R_' + arg.split('_')[0] + '_mean'
        # construct properly
        for original_key in ('R_urms_mean', 'R_half_mean', 'R_fifth_mean',
                              'R_tenth_mean'):
            if original_key.lower() == arg_lo:
                if original_key in tracking:
                    return np.full(nw, float(tracking[original_key])), \
                           'constant_' + arg_lo
                else:
                    return np.full(nw, default_fallback), 'fallback'
    try:
        return np.full(nw, float(arg)), f'literal_{arg}'
    except ValueError:
        raise ValueError(f"Couldn't interpret --cpc_fit_rho_max='{arg}' "
                         f"or --snr_local_radius='{arg}'")

cpc_fit_rho_max_arr, cpc_fit_rho_max_source = resolve_R_array(
    cpc_fit_rho_max_arg, tracking, sub_idxs, nw, default_fallback=0.1)
snr_local_radius_arr, snr_local_radius_source = resolve_R_array(
    snr_local_radius_arg, tracking, sub_idxs, nw,
    default_fallback=rho_max_template)
print(f"cpc_fit_rho_max:  source={cpc_fit_rho_max_source}, "
      f"mean={cpc_fit_rho_max_arr.mean():.4f}")
print(f"snr_local_radius: source={snr_local_radius_source}, "
      f"mean={snr_local_radius_arr.mean():.4f}")

### load EVP ###
logger.info(f"Loading EVP file: {evp_file}")
evp = np.load(evp_file, allow_pickle=True)[()]
for k in ('evals_res', 'psi_right_evecs_res'):
    if k not in evp:
        raise KeyError(f"EVP file missing '{k}'.")
left_key = None
for k_try in ('psi_left_evecs_res', 'psi_left_evecs', 'psi_mleft_evecs_res',
              'mleft_res', 'mleft', 'left_evecs_res', 'left_evecs'):
    if k_try in evp:
        left_key = k_try
        break
if left_key is None:
    raise KeyError(f"No left eigenvectors found.")
print(f"EVP left-evec key: '{left_key}'")

evals_raw     = np.asarray(evp['evals_res'])
psi_right_raw = np.asarray(evp['psi_right_evecs_res'])
psi_left_raw  = np.asarray(evp[left_key])

sort_idxs = np.argsort(evals_raw.imag)
evals     = evals_raw[sort_idxs]
psi_right = psi_right_raw[sort_idxs]
psi_left  = psi_left_raw[sort_idxs]
Nphi_deal_evp, Nr_deal_evp = psi_right.shape[1], psi_right.shape[2]

### candidates ###
if cand_idxs_arg.lower() == 'auto':
    n_cand = min(n_candidate_modes, len(evals))
    candidate_mode_idxs = np.arange(n_cand, dtype=int)
else:
    candidate_mode_idxs = np.array(
        [int(x) for x in cand_idxs_arg.split(',') if x.strip() != ''],
        dtype=int)
    n_cand = len(candidate_mode_idxs)
print(f"\nCandidates (n={n_cand}):")
for j, c in enumerate(candidate_mode_idxs):
    print(f"  cand {j}: idx={c}, eval={evals[c]}")

### FFT cross-ref ###
fft_xref_evp_idx, target_eval, fft_in_candidates = None, None, None
if fft_file_arg.lower() != 'none':
    fft = np.load(fft_file_arg, allow_pickle=True)[()]
    target_eval = complex(np.asarray(fft['evals_re'])[fft_mode_idx],
                          np.asarray(fft['evals_im'])[fft_mode_idx])
    dists = np.abs(evals - target_eval)
    fft_xref_evp_idx = int(np.argmin(dists))
    fft_in_candidates = bool(fft_xref_evp_idx in candidate_mode_idxs)

### Dedalus setup ###
if Nphi_arg.lower() == 'auto':
    Nphi = int(round(Nphi_deal_evp / dealias))
else:
    Nphi = int(Nphi_arg)
if Nr_arg.lower() == 'auto':
    Nr = int(round(Nr_deal_evp / dealias))
else:
    Nr = int(Nr_arg)
print(f"\nNphi={Nphi}, Nr={Nr}")

dtype  = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist   = d3.Distributor(coords, dtype=dtype)
disk   = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                      dealias=dealias, dtype=dtype)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))
phi_1d = phi_g[:, 0]
r_1d   = r_g[0, :]
Nphi_deal, Nr_deal = len(phi_1d), len(r_1d)

r_mesh   = np.tile(r_1d[np.newaxis, :], (Nphi_deal, 1))
phi_mesh = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))
x_mesh   = r_mesh * np.cos(phi_mesh)
y_mesh   = r_mesh * np.sin(phi_mesh)

### Poisson solver ###
vort_rhs        = dist.Field(name='vort_rhs',   bases=disk)
psi_field       = dist.Field(name='psi_field',  bases=disk)
tau_psi         = dist.Field(name='tau_psi',    bases=disk.edge)
lift            = lambda A: d3.Lift(A, disk.derivative_basis(2), -1)

problem = d3.LBVP([psi_field, tau_psi], namespace=locals())
problem.add_equation("lap(psi_field) + lift(tau_psi) = vort_rhs")
problem.add_equation("psi_field(r=1) = 0")
poisson_solver = problem.build_solver()
print("Poisson solver built.")

def solve_poisson(rhs_g):
    vort_rhs.change_scales(dealias)
    vort_rhs['g'] = rhs_g
    poisson_solver.solve()
    psi_field.change_scales(dealias)
    return np.array(psi_field['g'])

### Laplacian helper ###
psi_lap_input = dist.Field(name='psi_lap_input', bases=disk)
def laplacian_of_real_grid(g_array):
    psi_lap_input.change_scales(dealias)
    psi_lap_input['g'] = g_array
    result = d3.lap(psi_lap_input).evaluate()
    result.change_scales(dealias)
    return np.array(result['g'])

### precompute omega_R_m ###
print("\nPrecomputing omega^R_m = lap(psi^R_m) ...")
omega_R = np.zeros((n_cand, Nphi_deal, Nr_deal), dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    omega_R_re = laplacian_of_real_grid(psi_right[c].real.copy())
    omega_R_im = laplacian_of_real_grid(psi_right[c].imag.copy())
    omega_R[j] = omega_R_re + 1j * omega_R_im

### parametric model definitions ###
def lamb_oseen(rho, omega_0, L):
    return omega_0 * np.exp(-(rho / L)**2)

def rankine_smooth(rho, omega_0, L):
    return omega_0 / np.sqrt(1.0 + (rho / L)**2)

def rankine_strict(rho, omega_0, L):
    out = np.where(rho < L, omega_0, omega_0 * L / np.maximum(rho, 1e-30))
    return out

_model_funcs = {
    'lamb_oseen'     : lamb_oseen,
    'rankine_smooth' : rankine_smooth,
    'rankine_strict' : rankine_strict,
}

### CPC-frame alpha-average machinery ###
n_alpha_avg = 32
alpha_grid_avg = np.linspace(0, 2*np.pi, n_alpha_avg, endpoint=False)
# use locator's rho grid for the alpha average (log-spaced + 0)
rho_grid_avg = rho_fit_grid.copy()
n_rho_avg    = len(rho_grid_avg)

phi_1d_wrap = np.append(phi_1d, 2*np.pi)

def alpha_average_at_frame(vort_g, r_CPC_t, phi_CPC_t):
    """Return alpha-averaged omega_bar(rho) on rho_grid_avg at this frame."""
    rho_md, alpha_md = np.meshgrid(rho_grid_avg, alpha_grid_avg, indexing='ij')
    x_CPC = r_CPC_t * np.cos(phi_CPC_t)
    y_CPC = r_CPC_t * np.sin(phi_CPC_t)
    x_lab = x_CPC + rho_md * np.cos(alpha_md)
    y_lab = y_CPC + rho_md * np.sin(alpha_md)
    r_lab = np.sqrt(x_lab**2 + y_lab**2)
    phi_lab = np.arctan2(y_lab, x_lab) % (2*np.pi)
    valid = r_lab <= r_1d[-1]

    vort_wrap = np.vstack([vort_g, vort_g[0:1, :]])
    interp = RegularGridInterpolator((phi_1d_wrap, r_1d), vort_wrap,
                                      method='linear', bounds_error=False,
                                      fill_value=np.nan)
    pts = np.column_stack([phi_lab.ravel(), r_lab.ravel()])
    vort_cpc = interp(pts).reshape(rho_md.shape)
    vort_cpc[~valid] = np.nan

    omega_bar = np.full(n_rho_avg, np.nan)
    for k in range(n_rho_avg):
        v = vort_cpc[k, :]
        vmask = np.isfinite(v)
        if vmask.sum() > 0:
            omega_bar[k] = float(np.mean(v[vmask]))
    return omega_bar

### CPC profile fit helper ###
def fit_cpc_profile(rho_arr, omega_arr, model_name, rho_max_fit,
                    init_L_guess):
    """
    Fit one of the parametric models to (rho_arr, omega_arr) on
    rho_arr <= rho_max_fit.  Returns (omega_0, L) or (nan, nan) if the
    fit fails.
    """
    mask = (rho_arr <= rho_max_fit) & np.isfinite(omega_arr)
    if mask.sum() < 4:
        return np.nan, np.nan, np.nan
    rho_data   = rho_arr[mask]
    omega_data = omega_arr[mask]
    func       = _model_funcs[model_name]
    init_omega = float(omega_data[0])
    init_L     = float(init_L_guess)
    p0 = [init_omega, init_L]
    bounds = ([1e-30,        1e-6],
              [10*abs(init_omega) + 1e-6, 2*max(rho_max_fit, init_L)])
    try:
        popt, _ = curve_fit(func, rho_data, omega_data, p0=p0, bounds=bounds,
                             maxfev=400)
        omega_0_fit, L_fit = float(popt[0]), float(popt[1])
        fit_resid = float(np.sqrt(np.mean(
            (omega_data - func(rho_data, omega_0_fit, L_fit))**2)))
        return omega_0_fit, L_fit, fit_resid
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan

### template builder from fitted params ###
def build_template_from_params(model_name, omega_0_fit, L_fit,
                                r_CPC_t, phi_CPC_t, rho_max_eval):
    """Build omega_T on the lab-frame Dedalus grid using the fitted profile.
    Zero beyond rho_max_eval (where the CPC has decayed to ~background)."""
    x_CPC = r_CPC_t * np.cos(phi_CPC_t)
    y_CPC = r_CPC_t * np.sin(phi_CPC_t)
    rho   = np.sqrt((x_mesh - x_CPC)**2 + (y_mesh - y_CPC)**2)
    if model_name == 'empirical':
        template = np.interp(rho.ravel(), rho_fit_grid, vort_radial_tavg,
                              right=0.0).reshape(rho.shape)
    else:
        func = _model_funcs[model_name]
        template = func(rho, omega_0_fit, L_fit)
        # zero out beyond rho_max_eval (or where parametric model gets noisy)
        template = np.where(rho <= rho_max_eval, template, 0.0)
    return template

### inner product (full disk, raw integral) ###
def inner_prod_disk(left_g, right_g):
    integrand = np.conj(left_g) * right_g
    phi_int   = np.sum(integrand, axis=0) * (2.0 * np.pi / Nphi_deal)
    val       = np.trapezoid(phi_int * r_1d, r_1d)
    return complex(val)

### biortho sanity ###
print("\nBiortho <psi_L_n, psi_R_n>:")
biortho_diag = np.zeros(n_cand, dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    biortho_diag[j] = inner_prod_disk(psi_left[c], psi_right[c])
    print(f"  cand {j} (idx {c}): "
          f"{biortho_diag[j].real:+.6e}{biortho_diag[j].imag:+.6e}j "
          f"(|.|={abs(biortho_diag[j]):.6e})")

### initial L guess for the parametric fit ###
if 'R_half_mean' in tracking:
    init_L_default = float(tracking['R_half_mean'])
elif 'R_urms_mean' in tracking:
    init_L_default = float(tracking['R_urms_mean']) / 2
else:
    init_L_default = 0.05
print(f"\nInit-L guess for per-frame fits: {init_L_default:.4f}")

### open HDF5 ###
f_h5  = h5py.File(hdf5_file, 'r')
t_all = f_h5['tasks/vort'].dims[0]['sim_time'][:]
if ws_sub.max() >= len(t_all):
    raise ValueError(f"ws index out of HDF5 range.")
vort_loader = dist.Field(name='vort', bases=disk)

### storage ###
A_RW_history          = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
a_history             = np.full(nw, np.nan + 0j,           dtype=np.complex128)
Delta_omega_history   = np.full(nw, np.nan + 0j,           dtype=np.complex128)
omega_T_norm2_hist    = np.full(nw, np.nan)
rho_T_omega_history   = np.full(nw, np.nan)
omega_residual_L2_ratio_hist = np.full(nw, np.nan)
omega_L2_hist         = np.full(nw, np.nan)
SNR_a_history         = np.full(nw, np.nan)
SNR_a_local_history   = np.full(nw, np.nan)
omega_0_fit_history   = np.full(nw, np.nan)
L_fit_history         = np.full(nw, np.nan)
fit_resid_history     = np.full(nw, np.nan)
C_history             = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
D_history             = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)

mid_idx = nw // 2
while mid_idx < nw and glitch_sub[mid_idx]:
    mid_idx += 1
snapshot_vort     = None
snapshot_template = None
snapshot_residual = None
snapshot_omega_bar= None
snapshot_omega_0  = None
snapshot_L        = None
snapshot_t        = None
snapshot_a        = None

prog_cad = max(1, nw // 50)
print(f"\nStarting per-frame fit + LSQ (nw={nw}, cpc_model={cpc_model}) ...")

for i in range(nw):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")
    if glitch_sub[i]:
        continue

    w = int(ws_sub[i])
    vort_loader.load_from_hdf5(f_h5, w)
    vort_loader.change_scales(dealias)
    vort_g = np.array(vort_loader['g'])

    r_CPC_t   = float(r_sub[i])
    phi_CPC_t = float(phi_sub[i])
    rho_max_fit_t = float(cpc_fit_rho_max_arr[i])

    # 1) alpha-averaged radial profile at this frame
    omega_bar_t = alpha_average_at_frame(vort_g, r_CPC_t, phi_CPC_t)

    # 2) parametric fit (or empirical fallback)
    if cpc_model == 'empirical':
        omega_0_t = peak0
        L_t       = np.nan  # not applicable
        fit_resid = np.nan
    else:
        omega_0_t, L_t, fit_resid = fit_cpc_profile(
            rho_grid_avg, omega_bar_t, cpc_model, rho_max_fit_t,
            init_L_guess=init_L_default)
        if not np.isfinite(omega_0_t):
            # fallback: use empirical template if fit failed
            print(f"  frame {i}: fit failed, using empirical template")
            cpc_model_this_frame = 'empirical'
        else:
            cpc_model_this_frame = cpc_model

    # 3) template on lab-frame grid (extends out to rho_max_template
    # so the LSQ has the full CPC streamfunction tails)
    if cpc_model == 'empirical' or not np.isfinite(omega_0_t):
        omega_T_g = np.interp(
            np.sqrt((x_mesh - r_CPC_t*np.cos(phi_CPC_t))**2
                  + (y_mesh - r_CPC_t*np.sin(phi_CPC_t))**2).ravel(),
            rho_fit_grid, vort_radial_tavg, right=0.0).reshape(x_mesh.shape)
    else:
        omega_T_g = build_template_from_params(
            cpc_model, omega_0_t, L_t, r_CPC_t, phi_CPC_t,
            rho_max_eval=rho_max_template)

    # 4) Poisson solves
    psi_g   = solve_poisson(vort_g)
    psi_T_g = solve_poisson(omega_T_g)

    # 5) psi-space inner products (for A_n and b, D)
    b_vec = np.zeros(n_cand, dtype=np.complex128)
    D_vec = np.zeros(n_cand, dtype=np.complex128)
    C_vec = np.zeros(n_cand, dtype=np.complex128)
    for j, c in enumerate(candidate_mode_idxs):
        b_vec[j] = inner_prod_disk(psi_left[c], psi_g)
        D_vec[j] = inner_prod_disk(psi_left[c], psi_T_g)
        C_vec[j] = inner_prod_disk(psi_T_g,     psi_right[c])

    # 6) omega-space effective fields
    coef_b_over_B = b_vec / biortho_diag
    coef_D_over_B = D_vec / biortho_diag
    omega_T_eff = omega_T_g.astype(np.complex128).copy()
    omega_eff   = vort_g.astype(np.complex128).copy()
    for j in range(n_cand):
        omega_T_eff -= coef_D_over_B[j] * omega_R[j]
        omega_eff   -= coef_b_over_B[j] * omega_R[j]

    # 7) closure
    Delta_omega = inner_prod_disk(omega_T_eff, omega_T_eff).real
    num_omega   = inner_prod_disk(omega_T_eff, omega_eff)
    if Delta_omega < 1e-30:
        a_t = 0.0 + 0j
    else:
        a_t = num_omega / Delta_omega

    A_t = (b_vec - a_t * D_vec) / biortho_diag

    # 8) residual
    omega_residual_g = vort_g.astype(np.complex128) - a_t * omega_T_g
    for j in range(n_cand):
        omega_residual_g -= A_t[j] * omega_R[j]
    omega_residual_L2_sq = inner_prod_disk(omega_residual_g,
                                            omega_residual_g).real
    omega_L2_sq = inner_prod_disk(vort_g, vort_g).real
    omega_T_norm2 = inner_prod_disk(omega_T_g, omega_T_g).real

    # 9) local-CPC L2 norms for SNR_a^local
    local_R = float(snr_local_radius_arr[i])
    x_CPC_t = r_CPC_t * np.cos(phi_CPC_t)
    y_CPC_t = r_CPC_t * np.sin(phi_CPC_t)
    local_mask_g = ((x_mesh - x_CPC_t)**2
                  + (y_mesh - y_CPC_t)**2 <= local_R**2).astype(np.float64)
    # < ω_T_eff, mask * ω_T_eff > and similarly for residual
    omega_T_eff_local_sq = inner_prod_disk(
        omega_T_eff, local_mask_g * omega_T_eff).real
    omega_residual_local_sq = inner_prod_disk(
        omega_residual_g, local_mask_g * omega_residual_g).real
    SNR_a_local = float(
        abs(a_t) * np.sqrt(max(omega_T_eff_local_sq, 0))
        / max(np.sqrt(max(omega_residual_local_sq, 0)), 1e-30))

    # store
    omega_T_norm2_hist[i]    = float(omega_T_norm2)
    rho_T_omega_history[i]   = float(abs(Delta_omega) / max(omega_T_norm2, 1e-30))
    omega_L2_hist[i]         = float(np.sqrt(max(omega_L2_sq, 0)))
    omega_residual_L2_ratio_hist[i] = float(
        np.sqrt(max(omega_residual_L2_sq, 0) / max(omega_L2_sq, 1e-30)))
    SNR_a_history[i]         = float(abs(a_t) * np.sqrt(abs(Delta_omega))
                                      / max(np.sqrt(max(omega_residual_L2_sq, 0)),
                                            1e-30))
    SNR_a_local_history[i]   = SNR_a_local
    omega_0_fit_history[i]   = omega_0_t if np.isfinite(omega_0_t) else np.nan
    L_fit_history[i]         = L_t       if np.isfinite(L_t)       else np.nan
    fit_resid_history[i]     = fit_resid if np.isfinite(fit_resid) else np.nan
    Delta_omega_history[i]   = Delta_omega
    a_history[i]             = a_t
    A_RW_history[:, i]       = A_t
    C_history[:, i]          = C_vec
    D_history[:, i]          = D_vec

    if i == mid_idx:
        snapshot_vort     = vort_g.copy()
        snapshot_template = (a_t.real * omega_T_g).copy()
        snapshot_residual = (vort_g - a_t.real * omega_T_g).copy()
        snapshot_omega_bar= omega_bar_t.copy()
        snapshot_omega_0  = omega_0_t
        snapshot_L        = L_t
        snapshot_t        = float(tw_sub[i])
        snapshot_a        = complex(a_t)
        snapshot_rho_fit_max = rho_max_fit_t

f_h5.close()
print("Per-frame loop done.")

### stats ###
def compute_amp_stats(A_hist):
    mean_a = np.full(n_cand, np.nan)
    std_a  = np.full(n_cand, np.nan)
    nval   = np.zeros(n_cand, dtype=int)
    for j in range(n_cand):
        fin = (np.isfinite(A_hist[j].real) & np.isfinite(A_hist[j].imag))
        nval[j] = int(fin.sum())
        if nval[j] > 0:
            a = np.abs(A_hist[j, fin])
            mean_a[j] = float(np.mean(a))
            std_a[j]  = float(np.std(a))
    return mean_a, std_a, nval

mean_abs_A, std_abs_A, n_valid_A = compute_amp_stats(A_RW_history)
winner_local   = int(np.nanargmax(mean_abs_A))
winner_evp_idx = int(candidate_mode_idxs[winner_local])

fin_snr_loc = np.isfinite(SNR_a_local_history)
print(f"\nEmpirical winner: cand {winner_local}, EVP idx {winner_evp_idx}")
print(f"  eigenvalue:  {evals[winner_evp_idx]}")
print(f"  <|A_RW|>:    {mean_abs_A[winner_local]:.4e}")
print(f"  std|A_RW|:   {std_abs_A[winner_local]:.4e}")

print(f"\nPer-candidate <|A_RW|>:")
for j, c in enumerate(candidate_mode_idxs):
    flag = " <-- WINNER" if j == winner_local else ""
    print(f"  cand {j} (idx {c}): <|A|>={mean_abs_A[j]:.4e}, "
          f"std={std_abs_A[j]:.4e}{flag}")

print(f"\nFit diagnostics:")
print(f"  omega_0(t): mean={np.nanmean(omega_0_fit_history):.4e}, "
      f"std={np.nanstd(omega_0_fit_history):.4e}")
print(f"  L(t):       mean={np.nanmean(L_fit_history):.4e}, "
      f"std={np.nanstd(L_fit_history):.4e}")
print(f"  RMS fit residual: mean={np.nanmean(fit_resid_history):.4e}")
print(f"  fit failures: {int(np.sum(~np.isfinite(omega_0_fit_history)))} of "
      f"{int(np.sum(~glitch_sub))} non-glitch frames")

print(f"\n|a|: mean={np.nanmean(np.abs(a_history)):.4f}, "
      f"<Re a>={np.nanmean(a_history.real):.4f}")
print(f"<|Im(a)|/|a|>: "
      f"{np.nanmean(np.abs(a_history.imag)/(np.abs(a_history) + 1e-30)):.3f}")
print(f"rho_T_omega: mean={np.nanmean(rho_T_omega_history):.4f}")
print(f"||omega_res||/||omega||: mean="
      f"{np.nanmean(omega_residual_L2_ratio_hist):.4f}")
print(f"SNR_a (global): mean={np.nanmean(SNR_a_history):.3f}")
print(f"SNR_a (local):  mean={np.nanmean(SNR_a_local_history):.3f}, "
      f"median={float(np.nanmedian(SNR_a_local_history)):.3f}")
snr_ok    = (SNR_a_local_history > 10) & fin_snr_loc
snr_marg  = (SNR_a_local_history > 3) & (SNR_a_local_history <= 10) & fin_snr_loc
snr_bad   = (SNR_a_local_history <= 3) & fin_snr_loc
print(f"  SNR_a_local buckets: >10: "
      f"{100*snr_ok.sum()/max(fin_snr_loc.sum(), 1):.1f}%, "
      f"3-10: {100*snr_marg.sum()/max(fin_snr_loc.sum(), 1):.1f}%, "
      f"<=3: {100*snr_bad.sum()/max(fin_snr_loc.sum(), 1):.1f}%")

### save ###
results = {
    'A_RW_history'             : A_RW_history,
    'a_history'                : a_history,
    'tw'                       : tw_sub,
    'ws'                       : ws_sub,
    'r_locs_used'              : r_sub,
    'phi_locs_used'            : phi_sub,
    'glitch_flags_used'        : glitch_sub,
    'peak_sub'                 : peak_sub,
    # candidates / winner
    'candidate_mode_idxs'      : candidate_mode_idxs,
    'candidate_evals'          : evals[candidate_mode_idxs],
    'mean_abs_A'               : mean_abs_A,
    'std_abs_A'                : std_abs_A,
    'n_valid_A'                : n_valid_A,
    'winner_local'             : winner_local,
    'winner_evp_idx'           : winner_evp_idx,
    # quality
    'Delta_omega_history'      : Delta_omega_history,
    'omega_T_norm2_hist'       : omega_T_norm2_hist,
    'rho_T_omega_history'      : rho_T_omega_history,
    'omega_residual_L2_ratio_hist': omega_residual_L2_ratio_hist,
    'omega_L2_hist'            : omega_L2_hist,
    'SNR_a_history'            : SNR_a_history,
    'SNR_a_local_history'      : SNR_a_local_history,
    'C_history'                : C_history,
    'D_history'                : D_history,
    'biortho_diag'             : biortho_diag,
    # parametric fit
    'cpc_model'                : cpc_model,
    'omega_0_fit_history'      : omega_0_fit_history,
    'L_fit_history'            : L_fit_history,
    'fit_resid_history'        : fit_resid_history,
    'cpc_fit_rho_max_arr'      : cpc_fit_rho_max_arr,
    'cpc_fit_rho_max_source'   : cpc_fit_rho_max_source,
    'snr_local_radius_arr'     : snr_local_radius_arr,
    'snr_local_radius_source'  : snr_local_radius_source,
    # FFT cross-ref
    'fft_file'                 : fft_file_arg if fft_file_arg.lower() != 'none' else None,
    'fft_xref_evp_idx'         : fft_xref_evp_idx,
    'fft_target_eval'          : target_eval,
    'fft_in_candidates'        : fft_in_candidates,
    # metadata
    'rho_fit_grid'             : rho_fit_grid,
    'vort_radial_tavg'         : vort_radial_tavg,
    'r_CPC_mean'               : r_CPC_mean,
    'peak0'                    : peak0,
    'mean_peak'                : mean_peak,
    'evals_sorted'             : evals,
    'Nphi'                     : Nphi,
    'Nr'                       : Nr,
    'output_suffix'            : output_suffix,
    # snapshot
    'snapshot_t'               : snapshot_t,
    'snapshot_a'               : snapshot_a,
    'snapshot_omega_0'         : snapshot_omega_0,
    'snapshot_L'               : snapshot_L,
    'snapshot_omega_bar'       : snapshot_omega_bar,
    'snapshot_rho_grid'        : rho_grid_avg,
    'snapshot_rho_fit_max'     : snapshot_rho_fit_max
                                  if snapshot_vort is not None else None,
}
print(f"\nSaving: {output_path}")
np.save(output_path, results)

### diagnostic plot ###
if plot_path is not None:
    print(f"Saving plot: {plot_path}")
    fig = plt.figure(figsize=(17, 17), constrained_layout=True)
    gs  = fig.add_gridspec(4, 4)

    cand_colors = plt.cm.tab10(np.linspace(0, 1, max(n_cand, 10)))[:n_cand]

    # Row 0: |A_RW(t)|, bar chart, a(t), SNR_local(t)
    ax = fig.add_subplot(gs[0, 0])
    for j, c in enumerate(candidate_mode_idxs):
        is_win = (j == winner_local)
        ax.plot(tw_sub, np.abs(A_RW_history[j]),
                color=cand_colors[j],
                lw=1.2 if is_win else 0.7,
                alpha=1.0 if is_win else 0.55,
                label=f'idx {c}' + (' (winner)' if is_win else ''))
    ax.set_xlabel('t'); ax.set_ylabel(r'$|A_\mathrm{RW}|$')
    ax.set_title(r'$|A_{RW}(t)|$  ($\omega$-cost + per-frame fit)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(np.arange(n_cand), mean_abs_A, color=cand_colors, alpha=0.85)
    ax.errorbar(np.arange(n_cand), mean_abs_A, yerr=std_abs_A,
                fmt='none', color='k', capsize=3, alpha=0.6)
    bars[winner_local].set_edgecolor('red'); bars[winner_local].set_linewidth(2)
    ax.set_xticks(np.arange(n_cand))
    ax.set_xticklabels([str(c) for c in candidate_mode_idxs])
    ax.set_xlabel('EVP idx'); ax.set_ylabel(r'$\langle|A|\rangle$ ($\pm$ std)')
    ax.set_title('Mean amplitude per candidate')
    ax.grid(True, axis='y', alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(tw_sub, a_history.real, color='C0', lw=0.6, label=r'Re $a(t)$')
    ax.plot(tw_sub, a_history.imag, color='C3', lw=0.6, label=r'Im $a(t)$')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8,
               label=r'$a=1$ (template-only)')
    ax.axhline(np.nanmean(a_history.real), color='C0', ls='--', lw=1.0,
               label=fr'$\langle$Re$\rangle = {np.nanmean(a_history.real):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$a(t)$')
    ax.set_title('CPC amplitude (joint LSQ)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(tw_sub, SNR_a_local_history, color='C1', lw=0.7,
            label=r'$\mathrm{SNR}_a^\mathrm{local}$')
    ax.plot(tw_sub, SNR_a_history, color='gray', lw=0.4, alpha=0.6,
            label=r'$\mathrm{SNR}_a$ (global, for ref.)')
    ax.axhline(10, color='green', ls=':', lw=0.9, label='reliable (>10)')
    ax.axhline(3,  color='red',   ls=':', lw=0.9, label='unreliable (<3)')
    ax.axhline(np.nanmean(SNR_a_local_history), color='C1', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(SNR_a_local_history):.2f}$')
    ax.set_yscale('log')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\mathrm{SNR}_a$')
    ax.set_title(r'$\mathrm{SNR}_a^\mathrm{local}$ (CPC-localized)')
    ax.legend(fontsize=6, loc='best'); ax.grid(True, which='both', alpha=0.3)

    # Row 1: rho_T, residual fraction, fit residual, fit params
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tw_sub, rho_T_omega_history, color='C2', lw=0.7)
    ax.axhline(np.nanmean(rho_T_omega_history), color='C2', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(rho_T_omega_history):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\rho_T^\omega$')
    ax.set_title(r'$\omega$-space template independence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(0.05, float(np.nanmax(rho_T_omega_history))) * 1.1])

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(tw_sub, omega_residual_L2_ratio_hist, color='C3', lw=0.7)
    ax.axhline(np.nanmean(omega_residual_L2_ratio_hist), color='C3',
               ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(omega_residual_L2_ratio_hist):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\|\omega_\mathrm{res}\|/\|\omega\|$')
    ax.set_title('Post-fit residual fraction (global)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(tw_sub, fit_resid_history, color='C5', lw=0.7)
    ax.axhline(np.nanmean(fit_resid_history), color='C5', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(fit_resid_history):.2e}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'fit RMS resid')
    ax.set_title(f'Parametric fit residual ({cpc_model})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(tw_sub, omega_0_fit_history, color='C0', lw=0.6,
            label=r'$\omega_0(t)$')
    ax.axhline(peak0, color='C2', ls=':', lw=1.0,
               label=fr'locator peak0 = {peak0:.2f}')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\omega_0(t)$', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax_L = ax.twinx()
    ax_L.plot(tw_sub, L_fit_history, color='C1', lw=0.6, label=r'$L(t)$')
    ax_L.set_ylabel(r'$L(t)$', color='C1')
    ax_L.tick_params(axis='y', labelcolor='C1')
    ax.set_title(f'Per-frame fit parameters ({cpc_model})')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_L.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # Row 2: large radial profile diagnostic spanning all 4 cols
    ax = fig.add_subplot(gs[2, 0:4])
    if snapshot_vort is not None and snapshot_omega_bar is not None:
        ax.plot(rho_grid_avg, snapshot_omega_bar, color='C0', lw=1.4,
                label=fr'snapshot $\alpha$-avg ($t={snapshot_t:.2f}$)')
        # parametric template at fitted params
        if cpc_model != 'empirical' and np.isfinite(snapshot_omega_0):
            template_profile = _model_funcs[cpc_model](
                rho_grid_avg, snapshot_omega_0, snapshot_L)
            ax.plot(rho_grid_avg, template_profile, color='C2', lw=2.0,
                    label=fr'fitted template '
                          fr'($\omega_0={snapshot_omega_0:.1f}$, '
                          fr'$L={snapshot_L:.3f}$)')
            if snapshot_a is not None:
                ax.plot(rho_grid_avg, snapshot_a.real * template_profile,
                        color='C2', ls='--', lw=1.0,
                        label=fr'$a\cdot$fitted template, '
                              fr'$a={snapshot_a.real:.3f}$')
                residual_profile = snapshot_omega_bar - (
                    snapshot_a.real * template_profile)
                ax.plot(rho_grid_avg, residual_profile, color='C3', lw=1.2,
                        label='snapshot - $a\\cdot$template')
        else:
            template_profile = np.interp(rho_grid_avg, rho_fit_grid,
                                          vort_radial_tavg, right=0.0)
            ax.plot(rho_grid_avg, template_profile, color='C2', lw=2.0,
                    label='empirical template (locator $t$-avg)')
            if snapshot_a is not None:
                ax.plot(rho_grid_avg, snapshot_a.real * template_profile,
                        color='C2', ls='--', lw=1.0,
                        label=fr'$a\cdot$template, $a={snapshot_a.real:.3f}$')
        # mark the fit range
        ax.axvline(snapshot_rho_fit_max, color='gray', ls=':', lw=1.0,
                   label=fr'fit range max = {snapshot_rho_fit_max:.3f}')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\rho$ (CPC-frame radial distance)')
        ax.set_ylabel(r'$\bar\omega$')
        ax.set_title(r'CPC-frame radial profile: snapshot $\alpha$-avg vs '
                     f'fitted per-frame template ({cpc_model})')
        ax.legend(fontsize=9, loc='best'); ax.grid(True, which='both',
                                                     alpha=0.3)

    # Row 3: polar snapshots and summary text
    ax = fig.add_subplot(gs[3, 0], projection='polar')
    if snapshot_vort is not None:
        vmax = float(np.max(np.abs(snapshot_vort)))
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_vort,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(fr'$\omega(t={snapshot_t:.2f})$' if snapshot_t else r'$\omega$')

    ax = fig.add_subplot(gs[3, 1], projection='polar')
    if snapshot_template is not None:
        vmax = float(np.max(np.abs(snapshot_vort))) if snapshot_vort is not None else 1
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_template,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(fr'$a\,\omega_T$  ($a={snapshot_a.real:.3f}$)'
                 if snapshot_a is not None else r'$a\,\omega_T$')

    ax = fig.add_subplot(gs[3, 2], projection='polar')
    if snapshot_residual is not None:
        vmax = float(np.max(np.abs(snapshot_vort))) if snapshot_vort is not None else 1
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_residual,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(r'$\omega - a\,\omega_T$')

    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    #text  = f"output_suffix: {output_suffix}\n\n"
    text  = f"cpc_model: {cpc_model}\n"
    text += f"fit_rho_max src: {cpc_fit_rho_max_source}\n"
    text += f"  mean: {cpc_fit_rho_max_arr.mean():.4f}\n"
    text += f"snr_local_R src: {snr_local_radius_source}\n"
    text += f"  mean: {snr_local_radius_arr.mean():.4f}\n\n"
    text += f"<r_CPC>:           {r_CPC_mean:.4f}\n"
    text += f"<rho_T^omega>:     {np.nanmean(rho_T_omega_history):.3f}\n"
    text += f"<Re a>:            {np.nanmean(a_history.real):.4f}\n"
    text += f"<|a|>:             {np.nanmean(np.abs(a_history)):.4f}\n"
    text += f"<SNR_a (local)>:   {np.nanmean(SNR_a_local_history):.2f}\n"
    text += f"med(SNR_a local):  {float(np.nanmedian(SNR_a_local_history)):.2f}\n"
    text += f"frac SNR_loc>10:   "\
            f"{100*snr_ok.sum()/max(fin_snr_loc.sum(), 1):.1f}%\n"
    text += f"frac SNR_loc<3:    "\
            f"{100*snr_bad.sum()/max(fin_snr_loc.sum(), 1):.1f}%\n\n"
    text += f"<omega_0(t)>:      {np.nanmean(omega_0_fit_history):.2f}\n"
    text += f"<L(t)>:            {np.nanmean(L_fit_history):.4f}\n"
    text += f"locator peak0:     {peak0:.2f}\n"
    text += f"<fit RMS resid>:   {np.nanmean(fit_resid_history):.2e}\n\n"
    text += f"Candidates ({n_cand}):\n"
    for j, c in enumerate(candidate_mode_idxs):
        flag = "*" if j == winner_local else " "
        text += (f" {flag}{c:>2}: <|A|>={mean_abs_A[j]:.3e}\n")
    text += f"\nWinner: cand {winner_local} (EVP {winner_evp_idx})\n"
    text += f"  eval = {evals[winner_evp_idx].real:+.4f}"
    text += f"{evals[winner_evp_idx].imag:+.4f}j\n"
    text += f"  <|A|> = {mean_abs_A[winner_local]:.4e}\n"
    if fft_xref_evp_idx is not None:
        text += f"\nFFT xref idx: {fft_xref_evp_idx}\n"
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=7, family='monospace', verticalalignment='top')

    fig.suptitle(f'Per-frame fit + hybrid LSQ subtracted RW amplitude '
                 f'({cpc_model})', fontsize=11)# — {output_suffix}', fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

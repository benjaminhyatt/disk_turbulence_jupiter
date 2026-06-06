"""
RW amplitude via CPC-template hybrid LSQ (Phase A.2 — v3).

Same decomposition as v2:
    psi(t) = a(t) psi_T(t) + sum_{m in S} A_m(t) psi^R_m + psi_res(t)
but the cost function for determining a is now in VORTICITY space:
    a = argmin || omega - a*omega_T - sum_m A_m(a) omega^R_m ||^2

The A_m amplitudes are still determined by the standard psi-space
biorthogonal projection, A_n = (b_n - a*D_n)/biortho_n, exactly as in
v2.  Substituting A_n(a) into the omega-space cost and minimizing
gives the closed form:

    a = <omega_T_eff, omega_eff> / ||omega_T_eff||^2

where
    omega_T_eff = omega_T - sum_m (D_m / biortho_m) omega^R_m
    omega_eff   = omega   - sum_m (b_m / biortho_m) omega^R_m

The omega^R_m are precomputed once at startup from omega^R_m = lap(psi^R_m)
via Dedalus.

Motivation: in psi-space the L2 cost is dominated by broad-tail features
(because Poisson-inverted fields are smooth) and the LSQ ends up "letting"
the eigenmodes absorb the CPC's broad tails, biasing a toward zero.  In
omega-space the cost is dominated by sharp peaks — the CPC's sharpest
feature — so the LSQ should be forced to fit the CPC core with a non-
trivial a.

Quality diagnostics (omega-space analogs of v2):
    Delta_omega = ||omega_T_eff||^2
    rho_T_omega = Delta_omega / ||omega_T||^2
    SNR_a       = |a| sqrt(|Delta_omega|) / ||omega_residual||

New diagnostic plot panel: at a representative frame, compare the
CPC-frame alpha-averaged vorticity profile to the template profile
(vort_radial_tavg).  If they diverge substantially, the empirical
template is the wrong shape and a literature profile (Rankine, etc.)
should be substituted.

Usage:
    process_rw_amplitude_subtracted_v3.py <hdf5_file> <evp_file> <tracking_file> [options]

Options:
    --n_candidate_modes=<int>    [default: 4]
    --candidate_mode_idxs=<str>  'auto' or comma-separated [default: auto]

    --fft_file=<str>             optional FFT cross-reference;
                                 'none' to skip [default: none]
    --fft_mode_idx=<int>         [default: 0]
    --match_tol=<float>          [default: 1e-6]

    --t_start=<str>              [default: tracking]
    --t_end=<str>                [default: tracking]

    --output=<str>               [default: auto]
    --plot=<str>                 [default: auto]
    --output_prefix=<str>        [default: processed_rw_amplitude_subtracted_v2]
    --Nphi=<str>                 [default: auto]
    --Nr=<str>                   [default: auto]
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.interpolate import RegularGridInterpolator
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read args ###
args = docopt(__doc__)
print(args)

hdf5_file        = args['<hdf5_file>']
evp_file         = args['<evp_file>']
tracking_file    = args['<tracking_file>']

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
print(f"output_suffix (from tracking_file): {output_suffix}")

def auto_or(arg, default_template):
    return default_template if arg.lower() == 'auto' else arg

output_path = auto_or(output_arg, f"{output_prefix}_{output_suffix}.npy")
if plot_arg.lower() in ('none', ''):
    plot_path = None
else:
    plot_path = auto_or(plot_arg, f"{output_prefix}_{output_suffix}.png")
print(f"output .npy: {output_path}")
print(f"plot path:   {plot_path}")

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
        raise KeyError(f"Tracking file missing '{key}' (needed for template).")
rho_fit_grid       = np.array(tracking['rho_fit_grid'],       dtype=float)
vort_radial_tavg   = np.array(tracking['vort_radial_tavg'],   dtype=float)
omega_peak_history = np.array(tracking['omega_peak_history'], dtype=float)
peak0              = float(vort_radial_tavg[0])
rho_max_template   = float(rho_fit_grid[-1])
mean_peak          = float(np.nanmean(omega_peak_history))
print(f"tracking: {n_total_track} frames, {n_glitch} glitches, "
      f"<r_CPC>={r_CPC_mean:.4f}")
print(f"CPC shape template: rho_max = {rho_max_template:.3f}, "
      f"peak0 = {peak0:.4f}")

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
    raise KeyError(f"No left eigenvectors found. Keys: {list(evp.keys())}")
print(f"EVP left-evec key: '{left_key}'")

evals_raw     = np.asarray(evp['evals_res'])
psi_right_raw = np.asarray(evp['psi_right_evecs_res'])
psi_left_raw  = np.asarray(evp[left_key])

sort_idxs = np.argsort(evals_raw.imag)
evals     = evals_raw[sort_idxs]
psi_right = psi_right_raw[sort_idxs]
psi_left  = psi_left_raw[sort_idxs]

Nphi_deal_evp, Nr_deal_evp = psi_right.shape[1], psi_right.shape[2]

### candidate set ###
if cand_idxs_arg.lower() == 'auto':
    n_cand = min(n_candidate_modes, len(evals))
    candidate_mode_idxs = np.arange(n_cand, dtype=int)
else:
    candidate_mode_idxs = np.array(
        [int(x) for x in cand_idxs_arg.split(',') if x.strip() != ''],
        dtype=int)
    n_cand = len(candidate_mode_idxs)
print(f"\nCandidate modes (post-sort, n_cand={n_cand}):")
for j, c in enumerate(candidate_mode_idxs):
    ev = evals[c]
    print(f"  cand {j}: idx={c}, eval={ev.real:+.6f}{ev.imag:+.6f}j")

### optional FFT cross-reference ###
fft_xref_evp_idx, target_eval, fft_in_candidates = None, None, None
if fft_file_arg.lower() != 'none':
    print(f"\nLoading FFT cross-reference: {fft_file_arg}")
    fft = np.load(fft_file_arg, allow_pickle=True)[()]
    target_eval = complex(np.asarray(fft['evals_re'])[fft_mode_idx],
                          np.asarray(fft['evals_im'])[fft_mode_idx])
    dists = np.abs(evals - target_eval)
    fft_xref_evp_idx = int(np.argmin(dists))
    if float(dists[fft_xref_evp_idx]) > match_tol:
        print(f"  WARNING: |EVP - FFT| > tol")
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
print(f"\nNphi={Nphi}, Nr={Nr}  (dealias={dealias})")

dtype  = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist   = d3.Distributor(coords, dtype=dtype)
disk   = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                      dealias=dealias, dtype=dtype)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))
phi_1d = phi_g[:, 0]
r_1d   = r_g[0, :]
Nphi_deal, Nr_deal = len(phi_1d), len(r_1d)
print(f"Dedalus grid: ({Nphi_deal}, {Nr_deal})")

r_mesh   = np.tile(r_1d[np.newaxis, :], (Nphi_deal, 1))
phi_mesh = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))
x_mesh   = r_mesh * np.cos(phi_mesh)
y_mesh   = r_mesh * np.sin(phi_mesh)

### Poisson solver (only for psi from omega; not for psi_T anymore) ###
# Wait, we DO still need psi_T for the psi-space inner products (b, D).
# Solver same as v2.
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

### Laplacian helper for computing omega_R from psi_R ###
psi_lap_input  = dist.Field(name='psi_lap_input',  bases=disk)
def laplacian_of_real_grid(g_array):
    """Apply d3.lap to a real-valued grid array, return real-valued grid."""
    psi_lap_input.change_scales(dealias)
    psi_lap_input['g'] = g_array
    result = d3.lap(psi_lap_input).evaluate()
    result.change_scales(dealias)
    return np.array(result['g'])

### precompute omega_R_m for each candidate ###
print("\nPrecomputing omega^R_m = lap(psi^R_m) for each candidate ...")
omega_R = np.zeros((n_cand, Nphi_deal, Nr_deal), dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    omega_R_re = laplacian_of_real_grid(psi_right[c].real.copy())
    omega_R_im = laplacian_of_real_grid(psi_right[c].imag.copy())
    omega_R[j] = omega_R_re + 1j * omega_R_im

### inner product (full disk, raw integral) ###
def inner_prod_disk(left_g, right_g):
    integrand = np.conj(left_g) * right_g
    phi_int   = np.sum(integrand, axis=0) * (2.0 * np.pi / Nphi_deal)
    val       = np.trapezoid(phi_int * r_1d, r_1d)
    return complex(val)

### CPC template vorticity ###
def cpc_template_vorticity(r_CPC_t, phi_CPC_t):
    x_CPC = r_CPC_t * np.cos(phi_CPC_t)
    y_CPC = r_CPC_t * np.sin(phi_CPC_t)
    rho   = np.sqrt((x_mesh - x_CPC)**2 + (y_mesh - y_CPC)**2)
    template = np.interp(rho.ravel(), rho_fit_grid, vort_radial_tavg, right=0.0)
    return template.reshape(rho.shape)

### biortho sanity ###
print("\nBiorthogonal sanity <psi_L_n, psi_R_n>:")
biortho_diag = np.zeros(n_cand, dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    biortho_diag[j] = inner_prod_disk(psi_left[c], psi_right[c])
    print(f"  cand {j} (idx {c}): "
          f"{biortho_diag[j].real:+.6e}{biortho_diag[j].imag:+.6e}j  "
          f"(|.|={abs(biortho_diag[j]):.6e})")

### open HDF5 ###
f_h5  = h5py.File(hdf5_file, 'r')
t_all = f_h5['tasks/vort'].dims[0]['sim_time'][:]
if ws_sub.max() >= len(t_all):
    raise ValueError(f"ws index out of HDF5 range.")
vort_loader = dist.Field(name='vort', bases=disk)

### storage ###
A_RW_history       = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
a_history          = np.full(nw, np.nan + 0j, dtype=np.complex128)
Delta_omega_history= np.full(nw, np.nan + 0j, dtype=np.complex128)
omega_T_norm2_hist = np.full(nw, np.nan)
rho_T_omega_history= np.full(nw, np.nan)
omega_residual_L2_ratio_hist = np.full(nw, np.nan)
omega_L2_hist      = np.full(nw, np.nan)
SNR_a_history      = np.full(nw, np.nan)
C_history          = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
D_history          = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)

mid_idx = nw // 2
while mid_idx < nw and glitch_sub[mid_idx]:
    mid_idx += 1
snapshot_vort     = None
snapshot_template = None
snapshot_residual = None
snapshot_t        = None
snapshot_a        = None

prog_cad = max(1, nw // 50)
print(f"\nStarting per-frame hybrid LSQ "
      f"(nw={nw}, n_cand={n_cand}, omega-space cost) ...")
for i in range(nw):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")
    if glitch_sub[i]:
        continue

    w = int(ws_sub[i])
    vort_loader.load_from_hdf5(f_h5, w)
    vort_loader.change_scales(dealias)
    vort_g = np.array(vort_loader['g'])

    # psi from sim omega
    psi_g = solve_poisson(vort_g)

    # template omega and its psi
    omega_T_g = cpc_template_vorticity(float(r_sub[i]), float(phi_sub[i]))
    psi_T_g   = solve_poisson(omega_T_g)

    # psi-space inner products (still needed for A_m and D, b)
    b_vec = np.zeros(n_cand, dtype=np.complex128)
    D_vec = np.zeros(n_cand, dtype=np.complex128)
    C_vec = np.zeros(n_cand, dtype=np.complex128)
    for j, c in enumerate(candidate_mode_idxs):
        b_vec[j] = inner_prod_disk(psi_left[c], psi_g)
        D_vec[j] = inner_prod_disk(psi_left[c], psi_T_g)
        C_vec[j] = inner_prod_disk(psi_T_g,     psi_right[c])

    # omega-space cost: build omega_T_eff and omega_eff
    coef_b_over_B = b_vec / biortho_diag
    coef_D_over_B = D_vec / biortho_diag
    omega_T_eff = omega_T_g.astype(np.complex128).copy()
    omega_eff   = vort_g.astype(np.complex128).copy()
    for j in range(n_cand):
        omega_T_eff -= coef_D_over_B[j] * omega_R[j]
        omega_eff   -= coef_b_over_B[j] * omega_R[j]

    # closure: a = <omega_T_eff, omega_eff> / ||omega_T_eff||^2
    Delta_omega = inner_prod_disk(omega_T_eff, omega_T_eff).real
    num_omega   = inner_prod_disk(omega_T_eff, omega_eff)
    if Delta_omega < 1e-30:
        a_t = 0.0 + 0j
    else:
        a_t = num_omega / Delta_omega

    # A_m via psi-space biortho
    A_t = (b_vec - a_t * D_vec) / biortho_diag

    # post-fit omega-space residual
    omega_residual_g = vort_g.astype(np.complex128) - a_t * omega_T_g
    for j in range(n_cand):
        omega_residual_g -= A_t[j] * omega_R[j]
    omega_residual_L2_sq = inner_prod_disk(omega_residual_g, omega_residual_g).real
    omega_L2_sq          = inner_prod_disk(vort_g, vort_g).real

    omega_T_norm2 = inner_prod_disk(omega_T_g, omega_T_g).real

    omega_T_norm2_hist[i]    = float(omega_T_norm2)
    rho_T_omega_history[i]   = float(abs(Delta_omega) / max(omega_T_norm2, 1e-30))
    omega_L2_hist[i]         = float(np.sqrt(max(omega_L2_sq, 0)))
    omega_residual_L2_ratio_hist[i] = float(
        np.sqrt(max(omega_residual_L2_sq, 0) / max(omega_L2_sq, 1e-30)))
    SNR_a_history[i] = float(abs(a_t) * np.sqrt(abs(Delta_omega))
                              / max(np.sqrt(max(omega_residual_L2_sq, 0)), 1e-30))

    Delta_omega_history[i] = Delta_omega
    a_history[i]           = a_t
    A_RW_history[:, i]     = A_t
    C_history[:, i]        = C_vec
    D_history[:, i]        = D_vec

    if i == mid_idx:
        snapshot_vort     = vort_g.copy()
        snapshot_template = (a_t.real * omega_T_g).copy()
        snapshot_residual = (vort_g - a_t.real * omega_T_g).copy()
        snapshot_t        = float(tw_sub[i])
        snapshot_a        = complex(a_t)

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

fin_a   = np.isfinite(a_history.real) & np.isfinite(a_history.imag)
fin_snr = np.isfinite(SNR_a_history)

print(f"\nEmpirical winner: cand {winner_local}, EVP idx {winner_evp_idx}")
print(f"  eigenvalue:  {evals[winner_evp_idx]}")
print(f"  <|A_RW|>:    {mean_abs_A[winner_local]:.4e}")
print(f"  std|A_RW|:   {std_abs_A[winner_local]:.4e}")

print(f"\nPer-candidate <|A_RW|>:")
for j, c in enumerate(candidate_mode_idxs):
    flag = " <-- WINNER" if j == winner_local else ""
    print(f"  cand {j} (idx {c}): <|A|> = {mean_abs_A[j]:.4e}, "
          f"std = {std_abs_A[j]:.4e}{flag}")

print(f"\nQuality diagnostics:")
print(f"  |a|:       mean = {np.nanmean(np.abs(a_history)):.4e}, "
      f"std = {np.nanstd(np.abs(a_history)):.4e}")
print(f"  Re(a):     mean = {np.nanmean(a_history.real):.4e}")
print(f"  Im(a)/|a|: mean = "
      f"{np.nanmean(np.abs(a_history.imag) / (np.abs(a_history) + 1e-30)):.3f}")
print(f"  rho_T_omega: mean = {np.nanmean(rho_T_omega_history):.4f}, "
      f"min = {np.nanmin(rho_T_omega_history):.4f}")
print(f"  ||omega_res||/||omega||: mean = "
      f"{np.nanmean(omega_residual_L2_ratio_hist):.4f}")
print(f"  SNR_a: mean = {np.nanmean(SNR_a_history):.3f}, "
      f"median = {float(np.nanmedian(SNR_a_history)):.3f}")
snr_ok    = (SNR_a_history > 10) & fin_snr
snr_marg  = (SNR_a_history > 3) & (SNR_a_history <= 10) & fin_snr
snr_bad   = (SNR_a_history <= 3) & fin_snr
print(f"  SNR_a buckets: >10: "
      f"{100*snr_ok.sum()/max(fin_snr.sum(), 1):.1f}%, "
      f"3-10: {100*snr_marg.sum()/max(fin_snr.sum(), 1):.1f}%, "
      f"<=3: {100*snr_bad.sum()/max(fin_snr.sum(), 1):.1f}%")

### compute snapshot alpha-averaged radial profile for diagnostic ###
print("\nComputing snapshot CPC-frame alpha-averaged profile ...")
if snapshot_vort is not None:
    n_rho_diag   = 80
    n_alpha_diag = 32
    rho_diag = np.logspace(np.log10(max(rho_fit_grid[1], 1e-3)),
                           np.log10(rho_max_template), n_rho_diag)
    rho_diag = np.concatenate([[0.0], rho_diag])
    alpha_diag = np.linspace(0, 2*np.pi, n_alpha_diag, endpoint=False)
    rho_md, alpha_md = np.meshgrid(rho_diag, alpha_diag, indexing='ij')

    x_CPC_snap = float(r_sub[mid_idx]) * np.cos(float(phi_sub[mid_idx]))
    y_CPC_snap = float(r_sub[mid_idx]) * np.sin(float(phi_sub[mid_idx]))
    x_lab = x_CPC_snap + rho_md * np.cos(alpha_md)
    y_lab = y_CPC_snap + rho_md * np.sin(alpha_md)
    r_lab = np.sqrt(x_lab**2 + y_lab**2)
    phi_lab = np.arctan2(y_lab, x_lab) % (2*np.pi)
    valid = r_lab <= r_1d[-1]

    phi_1d_wrap = np.append(phi_1d, 2*np.pi)
    vort_wrap = np.vstack([snapshot_vort, snapshot_vort[0:1, :]])
    interp = RegularGridInterpolator((phi_1d_wrap, r_1d), vort_wrap,
                                      method='linear', bounds_error=False,
                                      fill_value=np.nan)
    pts = np.column_stack([phi_lab.ravel(), r_lab.ravel()])
    vort_cpc = interp(pts).reshape(rho_md.shape)
    # mask out points outside disk
    vort_cpc[~valid] = np.nan

    snapshot_alpha_avg = np.full(n_rho_diag + 1, np.nan)
    for k in range(n_rho_diag + 1):
        v = vort_cpc[k, :]
        vmask = np.isfinite(v)
        if vmask.sum() > 0:
            snapshot_alpha_avg[k] = float(np.mean(v[vmask]))
else:
    rho_diag           = None
    snapshot_alpha_avg = None

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
    'candidate_mode_idxs'      : candidate_mode_idxs,
    'candidate_evals'          : evals[candidate_mode_idxs],
    'mean_abs_A'               : mean_abs_A,
    'std_abs_A'                : std_abs_A,
    'n_valid_A'                : n_valid_A,
    'winner_local'             : winner_local,
    'winner_evp_idx'           : winner_evp_idx,
    # omega-space quality
    'Delta_omega_history'      : Delta_omega_history,
    'omega_T_norm2_hist'       : omega_T_norm2_hist,
    'rho_T_omega_history'      : rho_T_omega_history,
    'omega_residual_L2_ratio_hist': omega_residual_L2_ratio_hist,
    'omega_L2_hist'            : omega_L2_hist,
    'SNR_a_history'            : SNR_a_history,
    'C_history'                : C_history,
    'D_history'                : D_history,
    'biortho_diag'             : biortho_diag,
    # FFT cross-ref
    'fft_file'                 : fft_file_arg if fft_file_arg.lower() != 'none' else None,
    'fft_xref_evp_idx'         : fft_xref_evp_idx,
    'fft_target_eval'          : target_eval,
    'fft_in_candidates'        : fft_in_candidates,
    # template & metadata
    'rho_fit_grid'             : rho_fit_grid,
    'vort_radial_tavg'         : vort_radial_tavg,
    'r_CPC_mean'               : r_CPC_mean,
    'peak0'                    : peak0,
    'mean_peak'                : mean_peak,
    'evals_sorted'             : evals,
    'Nphi'                     : Nphi,
    'Nr'                       : Nr,
    'output_suffix'            : output_suffix,
    'snapshot_rho_diag'        : rho_diag,
    'snapshot_alpha_avg'       : snapshot_alpha_avg,
    'snapshot_t'               : snapshot_t,
    'snapshot_a'               : snapshot_a,
}
print(f"\nSaving: {output_path}")
np.save(output_path, results)

### diagnostic plot ###
if plot_path is not None:
    print(f"Saving plot: {plot_path}")
    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    gs  = fig.add_gridspec(3, 4)

    cand_colors = plt.cm.tab10(np.linspace(0, 1, max(n_cand, 10)))[:n_cand]

    # (0,0): per-candidate |A_RW(t)|
    ax = fig.add_subplot(gs[0, 0])
    for j, c in enumerate(candidate_mode_idxs):
        is_win = (j == winner_local)
        ax.plot(tw_sub, np.abs(A_RW_history[j]),
                color=cand_colors[j],
                lw=1.2 if is_win else 0.7,
                alpha=1.0 if is_win else 0.55,
                label=f'idx {c}' + (' (winner)' if is_win else ''))
    ax.set_xlabel('t'); ax.set_ylabel(r'$|A_\mathrm{RW}|$')
    ax.set_title(r'$|A_{RW}(t)|$  (hybrid LSQ, $\omega$-cost)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # (0,1): bar chart
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

    # (0,2): a(t)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(tw_sub, a_history.real, color='C0', lw=0.6, label=r'Re $a(t)$')
    ax.plot(tw_sub, a_history.imag, color='C3', lw=0.6, label=r'Im $a(t)$')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8, label=r'$a=1$ (template-only)')
    ax.axhline(np.nanmean(a_history.real), color='C0', ls='--', lw=1.0,
               label=fr'$\langle$Re$\rangle = {np.nanmean(a_history.real):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$a(t)$')
    ax.set_title('CPC amplitude (joint LSQ)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (0,3): SNR_a(t)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(tw_sub, SNR_a_history, color='C1', lw=0.7)
    ax.axhline(10, color='green', ls=':', lw=0.9, label='reliable (>10)')
    ax.axhline(3,  color='red',   ls=':', lw=0.9, label='unreliable (<3)')
    ax.axhline(np.nanmean(SNR_a_history), color='C1', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(SNR_a_history):.2f}$')
    ax.set_yscale('log')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\mathrm{SNR}_a$')
    ax.set_title(r'$\mathrm{SNR}_a$  ($\omega$-space residual)')
    ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)

    # (1,0): rho_T_omega(t)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tw_sub, rho_T_omega_history, color='C2', lw=0.7)
    ax.axhline(np.nanmean(rho_T_omega_history), color='C2', ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(rho_T_omega_history):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\rho_T^\omega$')
    ax.set_title(r'$\omega$-space template independence')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(0.05, float(np.nanmax(rho_T_omega_history))) * 1.1])

    # (1,1): residual fraction
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(tw_sub, omega_residual_L2_ratio_hist, color='C3', lw=0.7)
    ax.axhline(np.nanmean(omega_residual_L2_ratio_hist), color='C3',
               ls='--', lw=1.0,
               label=fr'$\langle\rangle = {np.nanmean(omega_residual_L2_ratio_hist):.3f}$')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\|\omega_\mathrm{res}\|/\|\omega\|$')
    ax.set_title(r'Post-fit residual fraction ($\omega$-space)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,2) and (1,3): radial profile diagnostic (TEMPLATE vs FRAME)
    ax = fig.add_subplot(gs[1, 2:4])
    if snapshot_alpha_avg is not None and rho_diag is not None:
        # template profile (the actual template values at rho)
        template_profile = np.interp(rho_diag, rho_fit_grid, vort_radial_tavg,
                                      right=0.0)
        ax.plot(rho_diag, template_profile, color='C2', lw=2.0,
                label=r'template $\bar\omega(\rho)$ (from locator $t$-avg)')
        ax.plot(rho_diag, snapshot_alpha_avg, color='C0', lw=1.2,
                label=fr'snapshot $\alpha$-avg at $t={snapshot_t:.2f}$')
        if snapshot_a is not None:
            ax.plot(rho_diag, snapshot_a.real * template_profile,
                    color='C2', ls='--', lw=1.0,
                    label=fr'$a\cdot$template, $a={snapshot_a.real:.3f}$')
        residual_profile = snapshot_alpha_avg - (
            snapshot_a.real * template_profile if snapshot_a is not None
            else template_profile)
        ax.plot(rho_diag, residual_profile, color='C3', lw=1.0,
                label='snapshot - $a\\cdot$template')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\rho$ (CPC-frame radial distance)')
        ax.set_ylabel(r'$\bar\omega$')
        ax.set_title(r'CPC-frame radial profile: template vs snapshot $\alpha$-avg')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, which='both', alpha=0.3)

    # (2,0): snapshot omega
    ax = fig.add_subplot(gs[2, 0], projection='polar')
    if snapshot_vort is not None:
        vmax = float(np.max(np.abs(snapshot_vort)))
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_vort,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(fr'$\omega(t={snapshot_t:.2f})$' if snapshot_t else r'$\omega$')

    # (2,1): a * omega_T
    ax = fig.add_subplot(gs[2, 1], projection='polar')
    if snapshot_template is not None:
        vmax = float(np.max(np.abs(snapshot_vort))) if snapshot_vort is not None else 1
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_template,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(fr'$a\,\omega_T$  ($a={snapshot_a.real:.3f}$)'
                 if snapshot_a is not None else r'$a\,\omega_T$')

    # (2,2): residual omega - a*omega_T
    ax = fig.add_subplot(gs[2, 2], projection='polar')
    if snapshot_residual is not None:
        vmax = float(np.max(np.abs(snapshot_vort))) if snapshot_vort is not None else 1
        pcm = ax.pcolormesh(phi_mesh, r_mesh, snapshot_residual,
                             shading='auto', cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
    ax.set_title(r'$\omega - a\,\omega_T$')

    # (2,3): summary text
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    #text  = f"output_suffix: {output_suffix}\n"
    text  = f"<r_CPC>:           {r_CPC_mean:.4f}\n"
    text += f"||omega_T||^2 mn:  {np.nanmean(omega_T_norm2_hist):.3e}\n"
    text += f"<rho_T^omega>:     {np.nanmean(rho_T_omega_history):.4f}\n"
    text += f"<Re a>:            {np.nanmean(a_history.real):.4f}\n"
    text += f"<|a|>:             {np.nanmean(np.abs(a_history)):.4f}\n"
    text += f"<Im(a)/|a|>:       "\
            f"{np.nanmean(np.abs(a_history.imag)/(np.abs(a_history) + 1e-30)):.3f}\n"
    text += f"<SNR_a>:           {np.nanmean(SNR_a_history):.2f}\n"
    text += f"median(SNR_a):     {float(np.nanmedian(SNR_a_history)):.2f}\n"
    text += f"frac SNR>10:       "\
            f"{100*snr_ok.sum()/max(fin_snr.sum(), 1):.1f}%\n"
    text += f"frac SNR<3:        "\
            f"{100*snr_bad.sum()/max(fin_snr.sum(), 1):.1f}%\n"
    text += f"<|omega_res|/|omega|>: "\
            f"{np.nanmean(omega_residual_L2_ratio_hist):.3f}\n\n"
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

    fig.suptitle(f'Hybrid LSQ ($\\omega$-cost) subtracted RW amplitude', fontsize=11)# — '
                 #f'{output_suffix}', fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

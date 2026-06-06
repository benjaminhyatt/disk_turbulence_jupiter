"""
Masked RW amplitude projection — multi-candidate (Phase A.2).

Projects the simulation streamfunction onto multiple candidate EVP
eigenmodes under an annular tanh-smoothed mask that excludes the CPC
band. For each non-glitch frame and each candidate mode n:

    A_RW_n(t)  =  <psi_left_n, M(t) * psi(t)>
                 ---------------------------------
                 <psi_left_n, M_avg * psi_right_n>

where:
  - psi(t)        is the streamfunction from the simulation vorticity via
                  the Poisson inversion  lap(psi) = vort  with psi(r=1) = 0
                  (+ sign convention, per user's setup).
  - psi_{L/R}_n   are the left / right biorthogonal eigenvectors.
  - M(t)(r)       is an annular tanh-smoothed mask in lab-frame r,
                  centered on r_CPC(t), excluding [r_CPC(t) - R, r_CPC(t) + R].
  - M_avg(r)      is the same mask but centered on <r_CPC>; using a single
                  denominator separates time-averaged normalization from
                  frame-to-frame mask motion in the numerator.

The candidate set is, by default, the first --n_candidate_modes EVP
indices after sorting by imag(eval) ascending. An explicit candidate set
can be supplied via --candidate_mode_idxs. The empirical dominant mode
under the masked methodology is the candidate with the largest
< |A_RW_n| >.

An FFT file is OPTIONAL and used only for cross-reference (--fft_file).
It does NOT drive mode selection here — that was the circularity we
wanted to avoid.

Mode-mixing diagnostic:

    I_nm = <psi_left_n, M_avg * psi_right_m>

computed for n, m in the union of the candidates' neighborhoods
(±--n_nearby_modes). Per-candidate mode-mixing ratios
max_{m != n} |I_nm / I_nn| are reported as a confidence indicator.

Usage:
    process_rw_amplitude_masked_v1.py <hdf5_file> <evp_file> <tracking_file> [options]

Arguments:
    <hdf5_file>      simulation analysis HDF5 with task 'vort'
    <evp_file>       EVP results (.npy); must contain evals_res,
                     psi_right_evecs_res, and a left-evec key
                     (psi_left_evecs_res / mleft_res / etc.)
    <tracking_file>  locator output (.npy)

Options:
    --R_mask=<str>               scale to use for the mask half-width;
                                 'R_urms', 'R_half', 'R_fifth', 'R_tenth',
                                 or a literal float [default: R_urms]
    --mask_width=<float>         tanh transition width [default: 0.01]

    --n_candidate_modes=<int>    number of candidate modes to project onto
                                 (first N after argsort by imag(eval))
                                 [default: 4]
    --candidate_mode_idxs=<str>  override; comma-separated EVP indices
                                 (post-sort), or 'auto' [default: auto]
    --n_nearby_modes=<int>       half-window around each candidate for I_nm
                                 [default: 5]

    --fft_file=<str>             optional FFT file for cross-reference
                                 only; 'none' to skip [default: none]
    --fft_mode_idx=<int>         FFT row to use for cross-reference
                                 [default: 0]
    --match_tol=<float>          warn if FFT-EVP eigenvalue match exceeds
                                 this [default: 1e-6]

    --t_start=<str>              override start time; 'tracking' uses
                                 tracking window [default: tracking]
    --t_end=<str>                override end time; 'tracking' uses
                                 tracking window [default: tracking]

    --output=<str>               output .npy path; 'auto' uses suffix
                                 [default: auto]
    --plot=<str>                 plot path; 'auto' or 'none' [default: auto]
    --output_prefix=<str>        prefix used when 'auto'
                                 [default: processed_rw_amplitude_masked]
    --Nphi=<str>                 azimuthal resolution; 'auto' from EVP file
                                 [default: auto]
    --Nr=<str>                   radial resolution; 'auto' from EVP file
                                 [default: auto]
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read args ###
args = docopt(__doc__)
print(args)

hdf5_file              = args['<hdf5_file>']
evp_file               = args['<evp_file>']
tracking_file          = args['<tracking_file>']

R_mask_arg             = args['--R_mask']
mask_width             = float(args['--mask_width'])

n_candidate_modes      = int(args['--n_candidate_modes'])
cand_idxs_arg          = args['--candidate_mode_idxs']
n_nearby_modes         = int(args['--n_nearby_modes'])

fft_file_arg           = args['--fft_file']
fft_mode_idx           = int(args['--fft_mode_idx'])
match_tol              = float(args['--match_tol'])

t_start_arg            = args['--t_start']
t_end_arg              = args['--t_end']

output_arg             = args['--output']
plot_arg               = args['--plot']
output_prefix          = args['--output_prefix']
Nphi_arg               = args['--Nphi']
Nr_arg                 = args['--Nr']

dealias                = 3/2

### output_suffix derivation (from tracking_file basename) ###
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

### load tracking file ###
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
print(f"tracking: {n_total_track} frames, {n_glitch} glitches, "
      f"<r_CPC>={r_CPC_mean:.4f}, "
      f"t in [{tw_track[0]:.3f}, {tw_track[-1]:.3f}]")

### resolve R_mask ###
def resolve_R_mask(arg, tracking):
    arg_lo = arg.lower()
    name_map = {
        'r_urms'  : 'R_urms_mean',
        'r_half'  : 'R_half_mean',
        'r_fifth' : 'R_fifth_mean',
        'r_tenth' : 'R_tenth_mean',
    }
    if arg_lo in name_map:
        key = name_map[arg_lo]
        if key not in tracking:
            raise KeyError(f"Tracking file missing key '{key}' "
                           f"needed for --R_mask={arg}")
        return float(tracking[key]), arg_lo
    return float(arg), 'literal'

R_mask, R_mask_src = resolve_R_mask(R_mask_arg, tracking)
print(f"R_mask = {R_mask:.4f}  (source: {R_mask_src})")
print(f"mask_width = {mask_width}  (tanh transition)")

### resolve time window ###
if t_start_arg.lower() == 'tracking':
    t_start_eff = float(tw_track[0])
else:
    t_start_eff = float(t_start_arg)
if t_end_arg.lower() == 'tracking':
    t_end_eff = float(tw_track[-1])
else:
    t_end_eff = float(t_end_arg)
print(f"time window: [{t_start_eff:.3f}, {t_end_eff:.3f}]")

# subset tracking arrays to the requested window
in_window  = (tw_track >= t_start_eff - 1e-9) & (tw_track <= t_end_eff + 1e-9)
sub_idxs   = np.where(in_window)[0]
ws_sub     = ws_track[sub_idxs]
tw_sub     = tw_track[sub_idxs]
r_sub      = r_locs[sub_idxs]
phi_sub    = phi_locs[sub_idxs]
glitch_sub = glitch_flags[sub_idxs]
nw         = len(ws_sub)
print(f"processing nw={nw} frames in window")

### load EVP file ###
logger.info(f"Loading EVP file: {evp_file}")
evp = np.load(evp_file, allow_pickle=True)[()]

required_keys = ['evals_res', 'psi_right_evecs_res']
for k in required_keys:
    if k not in evp:
        raise KeyError(f"EVP file missing required key '{k}'. "
                       f"Available keys: {list(evp.keys())}")

# left evec key — try a few common spellings
left_key = None
for k_try in ('psi_mleft_evecs_res', 'psi_left_evecs',
              'mleft_res', 'mleft', 'left_evecs_res', 'left_evecs'):
    if k_try in evp:
        left_key = k_try
        break
if left_key is None:
    raise KeyError("No left eigenvectors found in EVP file. "
                   f"Available keys: {list(evp.keys())}")
print(f"EVP left-evec key: '{left_key}'")

evals_raw     = np.asarray(evp['evals_res'])
psi_right_raw = np.asarray(evp['psi_right_evecs_res'])
psi_left_raw  = np.asarray(evp[left_key])
print(f"EVP: {len(evals_raw)} modes, "
      f"right evec shape={psi_right_raw.shape}, "
      f"left  evec shape={psi_left_raw.shape}")

# sort by imag(eval) ascending so mode indices are in frequency order
sort_idxs   = np.argsort(evals_raw.imag)
evals       = evals_raw[sort_idxs]
psi_right   = psi_right_raw[sort_idxs]
psi_left    = psi_left_raw[sort_idxs]

Nphi_deal_evp, Nr_deal_evp = psi_right.shape[1], psi_right.shape[2]

### candidate mode set ###
if cand_idxs_arg.lower() == 'auto':
    n_cand = min(n_candidate_modes, len(evals))
    candidate_mode_idxs = np.arange(n_cand, dtype=int)
else:
    candidate_mode_idxs = np.array(
        [int(x) for x in cand_idxs_arg.split(',') if x.strip() != ''],
        dtype=int)
    n_cand = len(candidate_mode_idxs)
    if np.any(candidate_mode_idxs < 0) or np.any(candidate_mode_idxs >= len(evals)):
        raise IndexError(f"--candidate_mode_idxs out of range "
                         f"[0, {len(evals)}): {candidate_mode_idxs.tolist()}")

print(f"\nCandidate modes (post-sort, n_cand={n_cand}):")
for j, c in enumerate(candidate_mode_idxs):
    ev = evals[c]
    print(f"  cand {j}: idx={c}, eval={ev.real:+.6f}{ev.imag:+.6f}j")

### optional FFT cross-reference ###
fft_xref_evp_idx = None
target_eval      = None
fft_in_candidates = None
if fft_file_arg.lower() != 'none':
    print(f"\nLoading FFT file for cross-reference: {fft_file_arg}")
    fft = np.load(fft_file_arg, allow_pickle=True)[()]
    fft_evals_re = np.asarray(fft['evals_re'])
    fft_evals_im = np.asarray(fft['evals_im'])
    if fft_mode_idx < 0 or fft_mode_idx >= len(fft_evals_re):
        raise IndexError(f"--fft_mode_idx={fft_mode_idx} out of range "
                         f"[0, {len(fft_evals_re)}).")
    target_eval = complex(fft_evals_re[fft_mode_idx],
                          fft_evals_im[fft_mode_idx])
    dists       = np.abs(evals - target_eval)
    fft_xref_evp_idx = int(np.argmin(dists))
    match_dist  = float(dists[fft_xref_evp_idx])
    print(f"  FFT target eval = "
          f"{target_eval.real:+.6f}{target_eval.imag:+.6f}j")
    print(f"  nearest EVP idx = {fft_xref_evp_idx}, "
          f"|EVP - FFT| = {match_dist:.3e}")
    if match_dist > match_tol:
        print(f"  WARNING: match exceeds tol {match_tol:.1e}")
    fft_in_candidates = bool(fft_xref_evp_idx in candidate_mode_idxs)
    if fft_in_candidates:
        print(f"  FFT-identified mode IS in the candidate set "
              f"(local idx {int(np.where(candidate_mode_idxs == fft_xref_evp_idx)[0][0])}).")
    else:
        print(f"  NOTE: FFT-identified mode is NOT in the candidate set "
              f"(only used for reference).")
else:
    print(f"\nNo FFT file supplied — skipping cross-reference.")

### Dedalus setup (real, mirroring the simulation) ###
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
Nphi_deal = len(phi_1d)
Nr_deal   = len(r_1d)
print(f"Dedalus grid: ({Nphi_deal}, {Nr_deal})")
if (Nphi_deal, Nr_deal) != (Nphi_deal_evp, Nr_deal_evp):
    raise ValueError(
        f"Dedalus grid ({Nphi_deal}, {Nr_deal}) != "
        f"EVP grid ({Nphi_deal_evp}, {Nr_deal_evp})")

r_mesh   = np.tile(r_1d[np.newaxis, :], (Nphi_deal, 1))
phi_mesh = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))

### Poisson solver: lap(psi) = vort, psi(r=1) = 0 ###
vort     = dist.Field(name='vort', bases=disk)
psi      = dist.Field(name='psi',  bases=disk)
tau_psi  = dist.Field(name='tau_psi', bases=disk.edge)
tau_psi_2 = dist.Field(name='tau_psi_2')
lift     = lambda A, n: d3.Lift(A, disk, n)

problem = d3.LBVP([psi, tau_psi, tau_psi_2], namespace=locals())
problem.add_equation("lap(psi) + lift(tau_psi, -1) + tau_psi_2 = vort")
problem.add_equation("psi(r=1) = 0")
problem.add_equation("integ(psi) = 0")
poisson_solver = problem.build_solver()
print("Poisson solver built.")

### mask + inner-product helpers ###
def compute_mask(r_CPC_t, R, w):
    """Annular tanh-smoothed exclusion mask centered on r_CPC_t.
    M = 0 inside [r_CPC - R, r_CPC + R], M = 1 outside, tanh transition w."""
    return 1.0 - 0.5 * (
        np.tanh((r_mesh - r_CPC_t + R) / w)
      - np.tanh((r_mesh - r_CPC_t - R) / w))

def inner_prod_disk(left_g, weight_g, right_g):
    """
    < left, weight * right > = int_disk conj(left)*weight*right dA.

    All inputs are (Nphi_deal, Nr_deal) numpy arrays. `left` and `right`
    may be complex; `weight` is typically real (the mask). Uses Fourier-
    uniform sum in phi and trapezoidal-with-r-weight in r.
    """
    integrand = np.conj(left_g) * weight_g * right_g
    phi_int   = np.sum(integrand, axis=0) * (2.0 * np.pi / Nphi_deal)
    val       = np.trapz(phi_int * r_1d, r_1d)
    return complex(val)

### time-averaged mask & per-candidate denominators (precomputed) ###
mask_avg_g = compute_mask(r_CPC_mean, R_mask, mask_width)
denoms = np.zeros(n_cand, dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    denoms[j] = inner_prod_disk(psi_left[c], mask_avg_g, psi_right[c])
print(f"\nPer-candidate denominators <psi_L_n, M_avg * psi_R_n>:")
for j, c in enumerate(candidate_mode_idxs):
    print(f"  cand {j} (idx {c}): "
          f"denom = {denoms[j].real:+.4e}{denoms[j].imag:+.4e}j, "
          f"|denom| = {abs(denoms[j]):.4e}")

### open HDF5 ###
f_h5  = h5py.File(hdf5_file, 'r')
t_all = f_h5['tasks/vort'].dims[0]['sim_time'][:]
if ws_sub.max() >= len(t_all):
    raise ValueError(f"ws index {ws_sub.max()} out of HDF5 range "
                     f"[0, {len(t_all)})")

### per-frame projection loop (over all candidates) ###
A_RW_history     = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
A_RW_num_history = np.full((n_cand, nw), np.nan + 0j, dtype=np.complex128)
prog_cad = max(1, nw // 50)

print(f"\nStarting per-frame projection (nw={nw}, n_cand={n_cand}) ...")
for i in range(nw):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")
    if glitch_sub[i]:
        continue

    w = int(ws_sub[i])
    vort.load_from_hdf5(f_h5, w)
    poisson_solver.solve()
    psi.change_scales(dealias)
    psi_g = np.array(psi['g'])

    r_CPC_t = float(r_sub[i])
    mask_t  = compute_mask(r_CPC_t, R_mask, mask_width)

    for j, c in enumerate(candidate_mode_idxs):
        num = inner_prod_disk(psi_left[c], mask_t, psi_g)
        A_RW_history[j, i]     = num / denoms[j]
        A_RW_num_history[j, i] = num

f_h5.close()
print("Per-frame projection loop done.")

### empirical winner via max <|A_RW|> ###
mean_abs_A = np.full(n_cand, np.nan)
std_abs_A  = np.full(n_cand, np.nan)
n_valid_A  = np.zeros(n_cand, dtype=int)
for j in range(n_cand):
    finite_j = np.isfinite(A_RW_history[j].real) & np.isfinite(A_RW_history[j].imag)
    n_valid_A[j] = int(finite_j.sum())
    if n_valid_A[j] > 0:
        abs_j         = np.abs(A_RW_history[j, finite_j])
        mean_abs_A[j] = float(np.mean(abs_j))
        std_abs_A[j]  = float(np.std(abs_j))
winner_local   = int(np.nanargmax(mean_abs_A))
winner_evp_idx = int(candidate_mode_idxs[winner_local])
print(f"\nEmpirical winner (max <|A_RW|>):")
print(f"  candidate {winner_local}, EVP idx {winner_evp_idx}")
print(f"  eigenvalue:  {evals[winner_evp_idx]}")
print(f"  <|A_RW|>:    {mean_abs_A[winner_local]:.4e}")
print(f"  std|A_RW|:   {std_abs_A[winner_local]:.4e}")
print(f"  valid frames: {n_valid_A[winner_local]} / {nw}")
if fft_xref_evp_idx is not None:
    if fft_xref_evp_idx == winner_evp_idx:
        print(f"  AGREES with FFT cross-reference (idx {fft_xref_evp_idx}).")
    else:
        print(f"  DIFFERS from FFT cross-reference (FFT idx {fft_xref_evp_idx}); "
              f"compare diagnostics carefully.")

### I_nm mode-mixing matrix (union of candidate neighborhoods, M_avg) ###
all_modes_set = set()
for c in candidate_mode_idxs:
    for k in range(max(0, c - n_nearby_modes),
                   min(len(evals), c + n_nearby_modes + 1)):
        all_modes_set.add(int(k))
mode_range = np.array(sorted(all_modes_set), dtype=int)
n_mix      = len(mode_range)
print(f"\nI_nm matrix: spans {n_mix} modes "
      f"[{mode_range.min()} .. {mode_range.max()}]")

I_nm = np.zeros((n_mix, n_mix), dtype=np.complex128)
I_nm_unmasked_diag = np.zeros(n_mix, dtype=np.complex128)
unit_g = np.ones_like(mask_avg_g)
for ii, ni in enumerate(mode_range):
    for jj, mj in enumerate(mode_range):
        I_nm[ii, jj] = inner_prod_disk(psi_left[ni], mask_avg_g,
                                        psi_right[mj])
    I_nm_unmasked_diag[ii] = inner_prod_disk(psi_left[ni], unit_g,
                                              psi_right[ni])

# candidate local positions within mode_range
candidate_local_in_mode_range = np.array(
    [int(np.where(mode_range == c)[0][0]) for c in candidate_mode_idxs],
    dtype=int)
winner_local_in_mode_range = candidate_local_in_mode_range[winner_local]

# per-candidate mode-mixing assessment
print(f"\nMode-mixing assessment per candidate:")
mixing_ratios = np.zeros(n_cand)
diag_masked_vals  = np.zeros(n_cand, dtype=np.complex128)
diag_unmasked_vals = np.zeros(n_cand, dtype=np.complex128)
for j, c in enumerate(candidate_mode_idxs):
    li = candidate_local_in_mode_range[j]
    diag_m = I_nm[li, li]
    diag_u = I_nm_unmasked_diag[li]
    diag_masked_vals[j]   = diag_m
    diag_unmasked_vals[j] = diag_u
    row = I_nm[li, :].copy()
    row[li] = 0
    off_max = float(np.max(np.abs(row)))
    ratio   = off_max / max(abs(diag_m), 1e-30)
    mixing_ratios[j] = ratio
    flag = "  <-- WINNER" if j == winner_local else ""
    warn = "  WARN: mixing > 0.1" if ratio > 0.1 else ""
    print(f"  cand {j} (idx {c}):  |I_nn|={abs(diag_m):.3e}  "
          f"|I_nn(masked)/I_nn(unmasked)|={abs(diag_m/diag_u):.4f}  "
          f"max|I_nm/I_nn|={ratio:.4f}{flag}{warn}")

### save results ###
results = {
    # primary
    'A_RW_history'      : A_RW_history,        # (n_cand, nw) complex
    'A_RW_num_history'  : A_RW_num_history,    # (n_cand, nw) complex
    'denoms'            : denoms,              # (n_cand,) complex
    'tw'                : tw_sub,
    'ws'                : ws_sub,
    'r_locs_used'       : r_sub,
    'phi_locs_used'     : phi_sub,
    'glitch_flags_used' : glitch_sub,
    # candidate set + winner
    'candidate_mode_idxs': candidate_mode_idxs,       # (n_cand,)
    'candidate_evals'   : evals[candidate_mode_idxs], # (n_cand,) complex
    'mean_abs_A'        : mean_abs_A,                 # (n_cand,) float
    'std_abs_A'         : std_abs_A,                  # (n_cand,) float
    'n_valid_A'         : n_valid_A,                  # (n_cand,) int
    'winner_local'      : winner_local,
    'winner_evp_idx'    : winner_evp_idx,
    'mixing_ratios'     : mixing_ratios,              # (n_cand,) float
    'diag_masked_vals'  : diag_masked_vals,           # (n_cand,) complex
    'diag_unmasked_vals': diag_unmasked_vals,         # (n_cand,) complex
    # mode-mixing diagnostic (M_avg-weighted)
    'I_nm_matrix'       : I_nm,                       # (n_mix, n_mix)
    'I_nn_unmasked_diag': I_nm_unmasked_diag,         # (n_mix,) complex
    'mode_range'        : mode_range,
    'candidate_local_in_mode_range': candidate_local_in_mode_range,
    'winner_local_in_mode_range'   : winner_local_in_mode_range,
    # FFT cross-reference (optional)
    'fft_file'          : fft_file_arg if fft_file_arg.lower() != 'none' else None,
    'fft_mode_idx'      : fft_mode_idx,
    'fft_xref_evp_idx'  : fft_xref_evp_idx,
    'fft_target_eval'   : target_eval,
    'fft_in_candidates' : fft_in_candidates,
    # mask settings
    'R_mask'            : R_mask,
    'R_mask_source'     : R_mask_src,
    'mask_width'        : mask_width,
    'r_CPC_mean'        : r_CPC_mean,
    # EVP metadata
    'evals_sorted'      : evals,
    # grid / setup
    'Nphi'              : Nphi,
    'Nr'                : Nr,
    'output_suffix'     : output_suffix,
}
print(f"\nSaving: {output_path}")
np.save(output_path, results)

### diagnostic plot ###
if plot_path is not None:
    print(f"Saving plot: {plot_path}")
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3)

    # color per candidate (use tab10)
    cand_colors = plt.cm.tab10(np.linspace(0, 1, max(n_cand, 10)))[:n_cand]

    # (0,0): |A_RW(t)| per candidate; winner highlighted
    ax = fig.add_subplot(gs[0, 0])
    for j, c in enumerate(candidate_mode_idxs):
        is_win = (j == winner_local)
        lw     = 1.2 if is_win else 0.7
        alpha  = 1.0 if is_win else 0.55
        label  = f'idx {c}' + ('  (winner)' if is_win else '')
        ax.plot(tw_sub, np.abs(A_RW_history[j]),
                color=cand_colors[j], lw=lw, alpha=alpha, label=label)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$|A_\mathrm{RW}|$')
    ax.set_title(r'Per-candidate $|A_{RW}(t)|$')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (0,1): bar chart of <|A_RW|> per candidate
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(np.arange(n_cand), mean_abs_A, color=cand_colors)
    ax.errorbar(np.arange(n_cand), mean_abs_A, yerr=std_abs_A,
                fmt='none', color='k', capsize=3, alpha=0.7)
    bars[winner_local].set_edgecolor('red')
    bars[winner_local].set_linewidth(2)
    ax.set_xticks(np.arange(n_cand))
    ax.set_xticklabels([str(c) for c in candidate_mode_idxs])
    ax.set_xlabel('EVP idx')
    ax.set_ylabel(r'$\langle|A_\mathrm{RW}|\rangle$  ($\pm$ std)')
    ax.set_title('Mean amplitude per candidate')
    ax.grid(True, axis='y', alpha=0.3)

    # (0,2): I_nm heatmap (log10 |I_nm|), candidates + winner marked
    ax = fig.add_subplot(gs[0, 2])
    pcm = ax.imshow(np.log10(np.abs(I_nm) + 1e-30),
                     origin='lower', cmap='viridis', aspect='equal',
                     extent=[mode_range[0]-0.5, mode_range[-1]+0.5,
                             mode_range[0]-0.5, mode_range[-1]+0.5])
    for c in candidate_mode_idxs:
        ax.axvline(c, color='r', ls='--', lw=0.6, alpha=0.5)
        ax.axhline(c, color='r', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(winner_evp_idx, color='cyan', ls='-', lw=1.2, alpha=0.9)
    ax.axhline(winner_evp_idx, color='cyan', ls='-', lw=1.2, alpha=0.9)
    if fft_xref_evp_idx is not None:
        ax.axvline(fft_xref_evp_idx, color='magenta', ls=':', lw=1.0, alpha=0.8)
        ax.axhline(fft_xref_evp_idx, color='magenta', ls=':', lw=1.0, alpha=0.8)
    ax.set_xlabel(r'EVP idx $m$')
    ax.set_ylabel(r'EVP idx $n$')
    ax.set_title(r'$\log_{10}|I_{nm}|$  (under $M_\mathrm{avg}$)')
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    # (1,0): mask shape M(r) at r_CPC_mean
    ax = fig.add_subplot(gs[1, 0])
    r_plot = np.linspace(0, 1, 800)
    M_plot = 1 - 0.5*(np.tanh((r_plot - r_CPC_mean + R_mask)/mask_width)
                     - np.tanh((r_plot - r_CPC_mean - R_mask)/mask_width))
    ax.plot(r_plot, M_plot, color='C2', lw=1.5,
            label=fr'$M_\mathrm{{avg}}(r;\,\langle r_\mathrm{{CPC}}\rangle={r_CPC_mean:.3f})$')
    ax.axvline(r_CPC_mean - R_mask, color='k', ls=':', lw=0.8)
    ax.axvline(r_CPC_mean + R_mask, color='k', ls=':', lw=0.8)
    ax.axvline(r_CPC_mean,          color='k', ls='--', lw=0.6)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$M(r)$')
    ax.set_title(fr'Mask shape — $R={R_mask:.4f}$, $w={mask_width}$')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): off-diagonal mixing for the WINNER's row in I_nm
    ax = fig.add_subplot(gs[1, 1])
    win_row = np.abs(I_nm[winner_local_in_mode_range, :])
    ax.bar(mode_range, win_row, color='C0', alpha=0.85)
    ax.axvline(winner_evp_idx, color='red', ls='--', lw=1.0,
               label=f'winner ($n={winner_evp_idx}$)')
    for c in candidate_mode_idxs:
        if c != winner_evp_idx:
            ax.axvline(c, color='gray', ls=':', lw=0.6, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel(r'EVP idx $m$')
    ax.set_ylabel(r'$|I_{nm}|$  (winner row)')
    ax.set_title(r"Mode-mixing — winner's row")
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # (1,2): summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    #text  = f"output_suffix: {output_suffix}\n"
    text = f"R_mask:     {R_mask:.4f}  ({R_mask_src})\n"
    text += f"mask_width: {mask_width}\n"
    text += f"<r_CPC>:    {r_CPC_mean:.4f}\n\n"
    text += f"Candidates ({n_cand}):\n"
    text += f"  {'idx':>4}  {'<|A|>':>10}  {'mixing':>8}\n"
    for j, c in enumerate(candidate_mode_idxs):
        flag = "*" if j == winner_local else " "
        text += (f"  {flag}{c:>3}  {mean_abs_A[j]:>10.3e}  "
                 f"{mixing_ratios[j]:>8.4f}\n")
    text += f"\nWinner: cand {winner_local} (EVP idx {winner_evp_idx})\n"
    text += f"  eval = {evals[winner_evp_idx].real:+.6f}"
    text += f"{evals[winner_evp_idx].imag:+.6f}j\n"
    text += f"  <|A|> = {mean_abs_A[winner_local]:.4e}\n"
    text += f"  std|A| = {std_abs_A[winner_local]:.4e}\n"
    text += f"  mixing = {mixing_ratios[winner_local]:.4f}\n"
    if fft_xref_evp_idx is not None:
        text += f"\nFFT xref EVP idx: {fft_xref_evp_idx}\n"
        text += f"  in candidates:  {fft_in_candidates}\n"
        text += f"  agrees w/ winner: {fft_xref_evp_idx == winner_evp_idx}\n"
    else:
        text += f"\nNo FFT cross-reference.\n"
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=8, family='monospace', verticalalignment='top')

    fig.suptitle(f'Masked RW amplitude (multi-candidate)', #— {output_suffix}',
                 fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

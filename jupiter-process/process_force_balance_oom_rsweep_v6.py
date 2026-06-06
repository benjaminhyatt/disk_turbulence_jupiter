"""
Order-of-magnitude force balance estimate for the CPC, as a function of radius.

Identifies the dominant RW automatically by cross-referencing the FFT projection
file against the EVP file: the FFT file already stores eigenvalues in
amplitude-sorted order, so row `fft_mode_idx` of the FFT file (default 0, i.e.
the most strongly projected mode) determines the eigenvalue, and the matching
mode is located in the EVP file (sorted by ascending imag(eval)) by
nearest-eigenvalue lookup. The corresponding psi eigenvector is the RW that
enters the force balance.

Loads the CPC tracking file to determine r_CPC, using the tracking-script-
provided glitch_flags to exclude bad frames. phi_CPC is then determined from
the azimuthal max of the EVP-mode-derived omega_RW field along the r = r_CPC
slice.

Compares the two signed radial gradients along phi = phi_CPC:

    dr(f)        = -gamma * r          (strictly negative)
    dr(omega_RW) = A * dr(nabla^2 psi_evec)(r, phi_CPC)   (signed)

Balance condition: dr(omega_RW)(r, phi_CPC) = gamma * r

Saves a results dictionary for use in a downstream summary plot.

gamma is inferred by default from the parameter substring 'gam_<...>' in the
tracking file path (falling back to the FFT and EVP file paths), using the
same str_to_float + nearest-of-known-values convention as other scripts in
this pipeline. The default output filenames likewise inherit an output_suffix
extracted from the tracking file path.

Usage:
    process_force_balance_oom_rsweep_v6.py <evp_file> <fft_file> <tracking_file> [options]

Arguments:
    <evp_file>       path to processed EVP .npy file
    <fft_file>       path to processed FFT projection .npy file (output of
                     process_fit_rossby_projections_fft_*.py)
    <tracking_file>  path to processed tracking .npy file

Options:
    --gamma=<str>           gamma value, or 'auto' to infer from filename
                            (rounded to nearest known gamma value) [default: auto]
    --Nphi=<str>            azimuthal resolution; if 'auto', derive from EVP eigenvector shape [default: auto]
    --Nr=<str>              radial resolution;    if 'auto', derive from EVP eigenvector shape [default: auto]
    --fft_mode_idx=<int>    FFT-file row to use as the dominant RW. Row 0 is the
                            largest-amplitude mode after the FFT script's
                            amplitude sort. [default: 0]
    --evp_mode_idx=<str>    Override the auto-detected EVP mode index (post-sort
                            by ascending imag(eval)). If 'auto', look it up via
                            the FFT file's eigenvalue. [default: auto]
    --match_tol=<float>     Warn if the |EVP eval - FFT eval| match exceeds this
                            absolute tolerance [default: 1e-6]
    --output=<str>          output figure filename, or 'auto' to use the
                            output_suffix inferred from <tracking_file>
                            [default: auto]
    --save_results=<str>    path to save results dictionary, or 'auto' to use
                            the output_suffix inferred from <tracking_file>
                            [default: auto]
    --output_prefix=<str>   prefix used when --output / --save_results are 'auto'
                            [default: force_balance_oom]
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read options ###
args = docopt(__doc__)
print(args)

evp_file       = args['<evp_file>']
fft_file       = args['<fft_file>']
tracking_file  = args['<tracking_file>']
gamma_arg      = args['--gamma']
Nphi_arg       = args['--Nphi']
Nr_arg         = args['--Nr']
fft_mode_idx   = int(args['--fft_mode_idx'])
evp_mode_arg   = args['--evp_mode_idx']
match_tol      = float(args['--match_tol'])
output_arg     = args['--output']
save_arg       = args['--save_results']
output_prefix  = args['--output_prefix']
dealias        = 3/2

### helpers for inferring gamma & output_suffix from filenames ###

# Same string -> float convention used throughout this pipeline
# (e.g. process_rossby_evp_v2.py, process_tracking_v7.py).
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])    # if str begins with format XdY (with d in {p, n})
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

# Union of gamma values used across the rest of the pipeline. Any gamma read
# from a filename is snapped to the nearest of these.
gamma_vals_known = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920,
                             2372, 2500, 3200))

def extract_output_suffix(file_path):
    """Strip path, extension, and known prefixes to recover the parameter suffix
    (the part containing 'Nphi_*_Nr_*_alpha_*_gam_*_eps_*_nu_*_kf_*')."""
    basename = file_path.split('/')[-1]
    if basename.endswith('.npy'):
        basename = basename[:-4]
    # try stripping any known prefix; otherwise return as-is
    for prefix in ('processed_tracking_',
                   'processed_rossby_projection_fft_',
                   'processed_rossby_projection_',
                   'processed_rossby_evp_',
                   'processed_profiles_',
                   'analysis_'):
        if prefix in basename:
            basename = basename.split(prefix, 1)[1]
            break
    return basename

def infer_gamma_from_path(file_path):
    """Find 'gam_<token>' in a filename and convert via str_to_float; return None
    on failure."""
    if 'gam_' not in file_path:
        return None
    try:
        gam_token = file_path.split('gam_')[1].split('_')[0].split('.')[0].split('/')[0]
        return str_to_float(gam_token)
    except Exception:
        return None

### resolve output_suffix (used for default output filenames) ###
output_suffix = extract_output_suffix(tracking_file)
print(f"output_suffix (from tracking_file): {output_suffix}")

### resolve gamma ###
if gamma_arg.lower() == 'auto':
    gamma_read = None
    gamma_src  = None
    for label, path in (('tracking_file', tracking_file),
                        ('fft_file',      fft_file),
                        ('evp_file',      evp_file)):
        v = infer_gamma_from_path(path)
        if v is not None:
            gamma_read = v
            gamma_src  = label
            break
    if gamma_read is None:
        raise ValueError("Unable to infer gamma from any input file path "
                         "(looked for 'gam_<token>' in tracking, fft, evp). "
                         "Pass --gamma=<float> explicitly.")
    gamma = float(gamma_vals_known[np.argmin(np.abs(gamma_vals_known - gamma_read))])
    print(f"gamma auto-inferred from {gamma_src}: "
          f"parsed={gamma_read:.4f} -> snapped to {gamma}")
else:
    gamma = float(gamma_arg)
    print(f"gamma set explicitly via --gamma: {gamma}")

### resolve default output filenames ###
if output_arg.lower() == 'auto':
    output = f"{output_prefix}_{output_suffix}.png"
else:
    output = output_arg
if save_arg.lower() == 'auto':
    save_results = f"{output_prefix}_results_{output_suffix}.npy"
else:
    save_results = save_arg
print(f"output figure path:  {output}")
print(f"save_results path:   {save_results}")

### load FFT file: dominant-mode eigenvalue & amplitude ###
logger.info("Loading FFT file: " + fft_file)
fft_data = np.load(fft_file, allow_pickle=True)[()]

# After process_fit_rossby_projections_fft_v5.py, modes are sorted by
# projdot_amp_extrema descending, so row 0 is the dominant projected mode.
try:
    fft_evals_re = np.asarray(fft_data['evals_re'])
    fft_evals_im = np.asarray(fft_data['evals_im'])
except KeyError as e:
    raise KeyError(f"FFT file missing required key {e}. Expected 'evals_re' "
                   "and 'evals_im' (saved by process_fit_rossby_projections_fft_v5.py).")

if fft_mode_idx < 0 or fft_mode_idx >= len(fft_evals_re):
    raise IndexError(f"--fft_mode_idx={fft_mode_idx} out of range "
                     f"[0, {len(fft_evals_re)}).")

target_eval = complex(fft_evals_re[fft_mode_idx], fft_evals_im[fft_mode_idx])
A_wave      = float(fft_data['projdot_amp_extrema'][fft_mode_idx])

print(f"\nFFT file row {fft_mode_idx} (amplitude-sorted):")
print(f"  eigenvalue: {target_eval.real:+.6f}{target_eval.imag:+.6f}j")
print(f"  projdot_amp_extrema (A_wave): {A_wave:.6f}")

# additional diagnostics if available
if 'fft_evp_match' in fft_data:
    evp_match_flag = bool(np.asarray(fft_data['fft_evp_match'])[fft_mode_idx])
    if not evp_match_flag:
        print("  WARNING: fft_evp_match is False for this row — the FFT-selected "
              "peak frequency did not fall within the configured window of the "
              "EVP eigenvalue. The mode lookup will still proceed by eigenvalue, "
              "but the chosen mode may not be the physical RW you expect.")
if 'fft_peak_harmonic_flag' in fft_data and 'fft_selected_idx' in fft_data:
    fsi = int(np.asarray(fft_data['fft_selected_idx'])[fft_mode_idx])
    if bool(np.asarray(fft_data['fft_peak_harmonic_flag'])[fft_mode_idx, fsi]):
        print("  WARNING: the FFT-selected peak is flagged as a possible "
              "harmonic of the dominant frequency for this mode.")

### load EVP file: sort, then locate the matching mode ###
logger.info("Loading EVP file: " + evp_file)
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

# auto-derive Nphi, Nr from EVP eigenvector shape (dealiased), if requested.
Nphi_deal_evp, Nr_deal_evp = evecs_sorted.shape[1], evecs_sorted.shape[2]
if Nphi_arg.lower() == 'auto':
    Nphi = int(round(Nphi_deal_evp / dealias))
    print(f"\n--Nphi auto-derived from EVP file: Nphi_deal={Nphi_deal_evp} -> Nphi={Nphi}")
else:
    Nphi = int(Nphi_arg)
if Nr_arg.lower() == 'auto':
    Nr = int(round(Nr_deal_evp / dealias))
    print(f"--Nr   auto-derived from EVP file: Nr_deal={Nr_deal_evp}   -> Nr={Nr}")
else:
    Nr = int(Nr_arg)

# locate the dominant EVP mode index
if evp_mode_arg.lower() == 'auto':
    # nearest-eigenvalue match against the FFT-file target eigenvalue
    dists           = np.abs(evals_sorted - target_eval)
    dominant_evp_idx = int(np.argmin(dists))
    match_dist      = float(dists[dominant_evp_idx])
    print(f"\nAuto-detected EVP mode via FFT-file eigenvalue lookup:")
    print(f"  EVP idx (sort_im_inc): {dominant_evp_idx}")
    print(f"  matched eigenvalue:    "
          f"{evals_sorted[dominant_evp_idx].real:+.6f}"
          f"{evals_sorted[dominant_evp_idx].imag:+.6f}j")
    print(f"  |EVP eval - FFT eval|: {match_dist:.3e}")
    if match_dist > match_tol:
        print(f"  WARNING: match distance exceeds --match_tol={match_tol:.1e}. "
              "Mode identification may be unreliable.")
    if len(dists) > 1:
        sorted_dists = np.sort(dists)
        print(f"  (2nd-nearest distance: {sorted_dists[1]:.3e})")
else:
    dominant_evp_idx = int(evp_mode_arg)
    print(f"\nEVP mode index set explicitly via --evp_mode_idx: idx={dominant_evp_idx}")
    print(f"  eigenvalue at that idx: "
          f"{evals_sorted[dominant_evp_idx].real:+.6f}"
          f"{evals_sorted[dominant_evp_idx].imag:+.6f}j")
    delta = abs(evals_sorted[dominant_evp_idx] - target_eval)
    print(f"  |EVP eval - FFT eval|:  {delta:.3e}  "
          f"(would auto-detect idx {int(np.argmin(np.abs(evals_sorted - target_eval)))})")

psi_evec_real = evecs_sorted[dominant_evp_idx].real
print(f"\nDominant EVP mode (idx={dominant_evp_idx}): "
      f"eval={evals_sorted[dominant_evp_idx].real:+.6f}"
      f"{evals_sorted[dominant_evp_idx].imag:+.6f}j")

# show nearby eigenvalues for context
n_show = min(5, len(evals_sorted))
print("\nNearby eigenvalues (post sort_im_inc):")
i_lo = max(0, dominant_evp_idx - n_show // 2)
i_hi = min(len(evals_sorted), i_lo + n_show)
for k in range(i_lo, i_hi):
    marker = '  <-- selected' if k == dominant_evp_idx else ''
    print(f"  idx={k}: "
          f"{evals_sorted[k].real:+.6f}{evals_sorted[k].imag:+.6f}j"
          f"{marker}")

### load tracking file and compute r_CPC (using tracker's glitch_flags) ###
logger.info("Loading tracking file: " + tracking_file)
tracking = np.load(tracking_file, allow_pickle=True)[()]

r_locs   = np.array(tracking['r_locs'],   dtype=float)
phi_locs = np.array(tracking['phi_locs'], dtype=float)

if 'glitch_flags' not in tracking:
    raise KeyError("Tracking file does not contain 'glitch_flags'. "
                   "Ensure the tracking script was run with a version that emits this key.")
glitch_flags = np.array(tracking['glitch_flags'], dtype=bool)
n_glitch     = int(np.sum(glitch_flags))

r_clean   = r_locs[~glitch_flags]
phi_clean = phi_locs[~glitch_flags]

if len(r_clean) == 0:
    raise RuntimeError("No non-glitch tracking frames found.")

r_CPC = float(np.mean(r_clean))

print(f"\nTracking file: {tracking_file}")
print(f"  Total points: {len(r_locs)},  glitch_flags excluded: {n_glitch}")
print(f"  r_CPC (mean over non-glitch frames) = {r_CPC:.4f}")

### Dedalus setup ###
coords     = d3.PolarCoordinates('phi', 'r')
dist       = d3.Distributor(coords, dtype=np.float64)
disk       = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                          dealias=dealias, dtype=np.float64)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))

er = dist.VectorField(coords, bases=disk)
er.change_scales(dealias)
er['g'][1] = 1.0

### grid axes ###
r_1d   = r_g[0, :]
phi_1d = phi_g[:, 0]

# consistency check: shape we set up must match EVP eigenvector shape
Nphi_deal_grid = phi_g.shape[0]
Nr_deal_grid   = r_g.shape[1]
if (Nphi_deal_grid, Nr_deal_grid) != (Nphi_deal_evp, Nr_deal_evp):
    raise ValueError(
        f"Dedalus grid shape ({Nphi_deal_grid}, {Nr_deal_grid}) does not match "
        f"EVP eigenvector shape ({Nphi_deal_evp}, {Nr_deal_evp}). "
        f"Check --Nphi/--Nr or the dealias factor.")

r_idx_CPC = int(np.argmin(np.abs(r_1d - r_CPC)))

### determine phi_CPC: argmax of -psi_evec along the r_CPC slice ###
phi_idx_psi = int(np.argmax(-psi_evec_real[:, r_idx_CPC]))
phi_CPC_psi = 2 * np.pi * phi_idx_psi / Nphi_deal_grid
print(f"  phi_CPC from psi_evec sign convention: {np.degrees(phi_CPC_psi):.2f} deg")

### build fields ###
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real

omega_RW = d3.lap(psi_RW).evaluate()
omega_RW.change_scales(dealias)

# Recompute phi_CPC from omega_RW (argmax along r_CPC slice). This is what the
# script downstream uses; we report both for cross-checking.
phi_idx = int(np.argmax(omega_RW['g'][:, r_idx_CPC]))
phi_CPC = 2 * np.pi * phi_idx / Nphi_deal_grid
print(f"  phi_CPC from omega_RW argmax:          {np.degrees(phi_CPC):.2f} deg")

domega_RW_dr = (er @ d3.grad(omega_RW)).evaluate()
domega_RW_dr.change_scales(dealias)

### radial profiles along phi_CPC slice ###
r_profile         = r_1d
domega_dr_profile = domega_RW_dr['g'][phi_idx, :]    # signed
grad_f_profile    = -gamma * r_profile                # signed: strictly negative
ratio_profile     = np.abs(domega_dr_profile) / (gamma * r_profile + 1e-30)

### find equilibrium radii where dr(omega_RW) = gamma * r ###
diff_profile = domega_dr_profile - gamma * r_profile
sign_changes = np.where(np.diff(np.sign(diff_profile)))[0]

r_eq_vals = []
for sc in sign_changes:
    r_lo, r_hi = r_profile[sc], r_profile[sc+1]
    f_lo, f_hi = diff_profile[sc], diff_profile[sc+1]
    r_eq = r_lo - f_lo * (r_hi - r_lo) / (f_hi - f_lo)
    r_eq_vals.append(r_eq)

r_idx_max  = int(np.argmax(np.abs(domega_dr_profile)))
r_max_grad = r_profile[r_idx_max]
max_grad   = np.abs(domega_dr_profile[r_idx_max])

### print results ###
print(f"\n{'='*65}")
print(f"Radial force balance scan — gamma={gamma:.0f}")
print(f"phi slice: {np.degrees(phi_1d[phi_idx]):.1f} deg")
print(f"{'='*65}")
print(f"{'r':>8}  {'dr(f)':>12}  {'dr(omRW)':>12}  {'ratio':>8}")
print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
step = max(1, len(r_profile) // 30)
for ri in range(0, len(r_profile), step):
    markers = []
    if abs(r_profile[ri] - r_CPC) < 2*(r_profile[1]-r_profile[0]):
        markers.append('r_CPC')
    if abs(r_profile[ri] - r_max_grad) < 2*(r_profile[1]-r_profile[0]):
        markers.append('max|dr(omega_RW)|')
    marker_str = '  <-- ' + ', '.join(markers) if markers else ''
    print(f"{r_profile[ri]:>8.4f}  {grad_f_profile[ri]:>12.4e}  "
          f"{domega_dr_profile[ri]:>12.4e}  {ratio_profile[ri]:>8.4f}{marker_str}")

print(f"\nEquilibrium radius/radii where dr(omega_RW) = gamma*r:")
if len(r_eq_vals) == 0:
    print("  None found in domain")
for r_eq in r_eq_vals:
    print(f"  r_eq = {r_eq:.4f}  (cf. r_CPC = {r_CPC:.4f}, "
          f"fractional diff = {abs(r_eq - r_CPC)/r_CPC:.3f})")

print(f"\nRadius of maximum |dr(omega_RW)|: r = {r_max_grad:.4f}")
print(f"  max |dr(omega_RW)| = {max_grad:.4e}")
print(f"  gamma * r_max      = {gamma * r_max_grad:.4e}")
print(f"  ratio at r_max     = {max_grad / (gamma * r_max_grad):.4f}")
print(f"\nAt r_CPC = {r_CPC:.4f}:")
print(f"  dr(f)        = {grad_f_profile[r_idx_CPC]:.4e}")
print(f"  dr(omega_RW) = {domega_dr_profile[r_idx_CPC]:.4e}")
print(f"  ratio        = {ratio_profile[r_idx_CPC]:.4f}")
print(f"{'='*65}")

### save results for summary plot ###
results = {
    'gamma'              : gamma,
    'r_CPC'              : r_CPC,
    'phi_CPC'            : phi_CPC,
    'phi_CPC_psi'        : phi_CPC_psi,
    'A_wave'             : A_wave,
    'r_eq'               : np.array(r_eq_vals),
    'r_max_grad'         : r_max_grad,
    'max_grad'           : max_grad,
    'ratio_at_rCPC'      : float(ratio_profile[r_idx_CPC]),
    'grad_f_at_rCPC'     : float(grad_f_profile[r_idx_CPC]),
    'domega_dr_at_rCPC'  : float(domega_dr_profile[r_idx_CPC]),
    # mode-selection metadata
    'evp_mode_idx'       : dominant_evp_idx,
    'evp_eval'           : evals_sorted[dominant_evp_idx],
    'fft_mode_idx'       : fft_mode_idx,
    'fft_eval_target'    : target_eval,
    'Nphi'               : Nphi,
    'Nr'                 : Nr,
    'n_glitch'           : n_glitch,
    'n_total'            : len(r_locs),
    'output_suffix'      : output_suffix,
    # full radial profiles for potential reuse
    'r_profile'          : r_profile,
    'domega_dr_profile'  : domega_dr_profile,
    'grad_f_profile'     : grad_f_profile,
    'ratio_profile'      : ratio_profile,
}
np.save(save_results, results)
print(f"Results saved to: {save_results}")

### figure ###
fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

# panel 1: signed radial gradients vs r
ax = axes[0]
ax.plot(r_profile, grad_f_profile,    color='C0', lw=1.5,
        label=r'$\partial_r f = -\gamma r$')
ax.plot(r_profile, domega_dr_profile, color='C1', lw=1.5,
        label=r'$\partial_r \omega_\mathrm{RW}$')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
for r_eq in r_eq_vals:
    if r_eq < 0.9:
        ax.axvline(r_eq, color='green', ls='-', lw=1.2,
                label=f'$r_{{\\rm eq}}={r_eq:.4f}$')
ax.set_xlabel(r'$r$')
ax.set_ylabel('Signed radial gradient')
ax.set_title(f'Force balance — $\\gamma={gamma:.0f}$, '
             f'$\\phi_{{\\rm CPC}}={np.degrees(phi_CPC):.1f}°$, $A={A_wave:.4f}$')
ax.set_xlim([0, 0.5])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# panel 2: ratio vs r
ax = axes[1]
ax.plot(r_profile, ratio_profile, color='C3', lw=1.5,
        label=r'$|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r$')
ax.axhline(1.0, color='k', ls='--', lw=1.0, label='Balance (ratio = 1)')
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
for r_eq in r_eq_vals:
    if r_eq < 0.9:
        ax.axvline(r_eq, color='green', ls='-', lw=1.2,
                label=f'$r_{{\\rm eq}}={r_eq:.4f}$')
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r$')
ax.set_title('Force balance ratio vs radius')
ax.set_xlim([0, 0.5])
ax.set_ylim([0, min(5, ratio_profile[r_profile > 0.02].max() * 1.2)])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.savefig(output, dpi=150)
plt.close(fig)
print(f"Figure saved: {output}")

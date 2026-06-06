"""
Three-term force balance estimate for the CPC, as a function of radius.

Extends the two-term v6 balance by adding the zonal-vorticity-gradient
contribution:

    dr(f)               = -gamma * r                              (signed)
    dr(omega_RW)        = A * dr(nabla^2 psi_evec)(r, phi_CPC)    (signed)
    dr(omega_zonal)(r)  = radial gradient of <vort>_{phi, t}       (signed)

Working three-term balance condition (placeholder, additive; analytical
derivation pending):

    dr(omega_RW)(r_CPC, phi_CPC) + dr(omega_zonal)(r_CPC) = gamma * r_CPC

The zonal vorticity is read from a 'processed_zonal_flow' .npy produced by
process_zonal_flow_v2.py. Because the m=0 vorticity is contaminated by the
CPC itself in a band of width +/- rho_window around r_CPC, the script
linearly interpolates omega_zonal(r) across that band using values at the
band edges, and reports the resulting central-difference estimate for the
gradient at r_CPC. The full profile (with band replaced by the linear
interp) is also constructed for the ratio plot.

EVP mode selection, FFT-driven RW identification, glitch handling, Nphi/Nr
auto-derivation, gamma auto-inference, and output-suffix conventions are
unchanged from v6.

Usage:
    process_force_balance_oom_rsweep_v7.py <evp_file> <fft_file> <tracking_file> <zonal_file> [options]

Arguments:
    <evp_file>       path to processed EVP .npy file
    <fft_file>       path to processed FFT projection .npy file
    <tracking_file>  path to processed tracking .npy file
    <zonal_file>     path to processed zonal-flow .npy file (from
                     process_zonal_flow_v2.py); must contain
                     'omega_zonal', 'r_1d', and (optionally) 'rho_window'

Options:
    --gamma=<str>           gamma value, or 'auto' to infer from filename [default: auto]
    --Nphi=<str>            azimuthal resolution; 'auto' from EVP file [default: auto]
    --Nr=<str>              radial resolution;    'auto' from EVP file [default: auto]
    --fft_mode_idx=<int>    FFT-file row to use as the dominant RW [default: 0]
    --evp_mode_idx=<str>    Override the auto-detected EVP mode index. If 'auto',
                            look it up via the FFT file's eigenvalue. [default: auto]
    --match_tol=<float>     Warn if the |EVP eval - FFT eval| match exceeds this
                            absolute tolerance [default: 1e-6]
    --rho_window=<str>      Half-width of the CPC-contaminated band used for the
                            linear interp; 'auto' to read from the zonal file
                            [default: auto]
    --output=<str>          output figure filename, or 'auto' [default: auto]
    --save_results=<str>    path to save results dict, or 'auto' [default: auto]
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
zonal_file     = args['<zonal_file>']
gamma_arg      = args['--gamma']
Nphi_arg       = args['--Nphi']
Nr_arg         = args['--Nr']
fft_mode_idx   = int(args['--fft_mode_idx'])
evp_mode_arg   = args['--evp_mode_idx']
match_tol      = float(args['--match_tol'])
rho_win_arg    = args['--rho_window']
output_arg     = args['--output']
save_arg       = args['--save_results']
output_prefix  = args['--output_prefix']
dealias        = 3/2

### helpers ###
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

gamma_vals_known = np.array((0, 30, 85, 240, 400, 675, 920, 950, 1200, 1920,
                             2372, 2500, 3200))

def extract_output_suffix(file_path):
    basename = file_path.split('/')[-1]
    if basename.endswith('.npy'):
        basename = basename[:-4]
    for prefix in ('processed_tracking_',
                   'processed_zonal_flow_',
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
    if 'gam_' not in file_path:
        return None
    try:
        gam_token = file_path.split('gam_')[1].split('_')[0].split('.')[0].split('/')[0]
        return str_to_float(gam_token)
    except Exception:
        return None

### resolve output_suffix and gamma ###
output_suffix = extract_output_suffix(tracking_file)
print(f"output_suffix (from tracking_file): {output_suffix}")

if gamma_arg.lower() == 'auto':
    gamma_read = None
    gamma_src  = None
    for label, path in (('tracking_file', tracking_file),
                        ('zonal_file',    zonal_file),
                        ('fft_file',      fft_file),
                        ('evp_file',      evp_file)):
        v = infer_gamma_from_path(path)
        if v is not None:
            gamma_read = v
            gamma_src  = label
            break
    if gamma_read is None:
        raise ValueError("Unable to infer gamma from any input file path. "
                         "Pass --gamma=<float> explicitly.")
    gamma = float(gamma_vals_known[np.argmin(np.abs(gamma_vals_known - gamma_read))])
    print(f"gamma auto-inferred from {gamma_src}: "
          f"parsed={gamma_read:.4f} -> snapped to {gamma}")
else:
    gamma = float(gamma_arg)

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

### load FFT file ###
logger.info("Loading FFT file: " + fft_file)
fft_data = np.load(fft_file, allow_pickle=True)[()]
fft_evals_re = np.asarray(fft_data['evals_re'])
fft_evals_im = np.asarray(fft_data['evals_im'])
if fft_mode_idx < 0 or fft_mode_idx >= len(fft_evals_re):
    raise IndexError(f"--fft_mode_idx={fft_mode_idx} out of range "
                     f"[0, {len(fft_evals_re)}).")
target_eval = complex(fft_evals_re[fft_mode_idx], fft_evals_im[fft_mode_idx])
A_wave      = float(fft_data['projdot_amp_extrema'][fft_mode_idx])
print(f"\nFFT file row {fft_mode_idx}: eval="
      f"{target_eval.real:+.6f}{target_eval.imag:+.6f}j,  A_wave={A_wave:.6f}")

if 'fft_evp_match' in fft_data:
    if not bool(np.asarray(fft_data['fft_evp_match'])[fft_mode_idx]):
        print("  WARNING: fft_evp_match=False for this row.")
if 'fft_peak_harmonic_flag' in fft_data and 'fft_selected_idx' in fft_data:
    fsi = int(np.asarray(fft_data['fft_selected_idx'])[fft_mode_idx])
    if bool(np.asarray(fft_data['fft_peak_harmonic_flag'])[fft_mode_idx, fsi]):
        print("  WARNING: selected FFT peak flagged as possible harmonic.")

### load EVP file ###
logger.info("Loading EVP file: " + evp_file)
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

Nphi_deal_evp, Nr_deal_evp = evecs_sorted.shape[1], evecs_sorted.shape[2]
if Nphi_arg.lower() == 'auto':
    Nphi = int(round(Nphi_deal_evp / dealias))
else:
    Nphi = int(Nphi_arg)
if Nr_arg.lower() == 'auto':
    Nr = int(round(Nr_deal_evp / dealias))
else:
    Nr = int(Nr_arg)
print(f"Nphi={Nphi}, Nr={Nr}")

if evp_mode_arg.lower() == 'auto':
    dists           = np.abs(evals_sorted - target_eval)
    dominant_evp_idx = int(np.argmin(dists))
    match_dist      = float(dists[dominant_evp_idx])
    print(f"Auto-detected EVP idx={dominant_evp_idx}, "
          f"|EVP - FFT| = {match_dist:.3e}")
    if match_dist > match_tol:
        print(f"  WARNING: match distance exceeds --match_tol={match_tol:.1e}.")
else:
    dominant_evp_idx = int(evp_mode_arg)
    print(f"EVP idx set explicitly: {dominant_evp_idx}")

psi_evec_real = evecs_sorted[dominant_evp_idx].real
print(f"Dominant EVP mode eval: "
      f"{evals_sorted[dominant_evp_idx].real:+.6f}"
      f"{evals_sorted[dominant_evp_idx].imag:+.6f}j")

### load tracking file ###
logger.info("Loading tracking file: " + tracking_file)
tracking = np.load(tracking_file, allow_pickle=True)[()]

r_locs   = np.array(tracking['r_locs'],   dtype=float)
phi_locs = np.array(tracking['phi_locs'], dtype=float)
if 'glitch_flags' not in tracking:
    raise KeyError("Tracking file missing 'glitch_flags'.")
glitch_flags = np.array(tracking['glitch_flags'], dtype=bool)
n_glitch     = int(np.sum(glitch_flags))
r_clean      = r_locs[~glitch_flags]
if len(r_clean) == 0:
    raise RuntimeError("No non-glitch tracking frames found.")
r_CPC = float(np.mean(r_clean))
print(f"r_CPC={r_CPC:.4f}  (glitches excluded: {n_glitch}/{len(r_locs)})")

### load zonal-flow file ###
logger.info("Loading zonal file: " + zonal_file)
zonal = np.load(zonal_file, allow_pickle=True)[()]
omega_zonal_full = np.asarray(zonal['omega_zonal'])
r_1d_zonal       = np.asarray(zonal['r_1d'])

if rho_win_arg.lower() == 'auto':
    if 'rho_window' in zonal:
        rho_window = float(zonal['rho_window'])
        print(f"rho_window auto-read from zonal_file: {rho_window:.3f}")
    else:
        rho_window = 0.3
        print(f"rho_window not in zonal_file; using default {rho_window:.3f}")
else:
    rho_window = float(rho_win_arg)
    print(f"rho_window set explicitly: {rho_window:.3f}")

### Dedalus setup ###
coords     = d3.PolarCoordinates('phi', 'r')
dist       = d3.Distributor(coords, dtype=np.float64)
disk       = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                          dealias=dealias, dtype=np.float64)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))
er = dist.VectorField(coords, bases=disk)
er.change_scales(dealias)
er['g'][1] = 1.0

r_1d   = r_g[0, :]
phi_1d = phi_g[:, 0]

Nphi_deal_grid = phi_g.shape[0]
Nr_deal_grid   = r_g.shape[1]
if (Nphi_deal_grid, Nr_deal_grid) != (Nphi_deal_evp, Nr_deal_evp):
    raise ValueError(
        f"Dedalus grid shape ({Nphi_deal_grid}, {Nr_deal_grid}) does not match "
        f"EVP eigenvector shape ({Nphi_deal_evp}, {Nr_deal_evp}).")
# also check zonal file shares grid
if r_1d.shape != r_1d_zonal.shape or not np.allclose(r_1d, r_1d_zonal, atol=1e-10):
    raise ValueError("Radial grid in zonal_file does not match the EVP/Dedalus grid. "
                     "Make sure the zonal-flow run used the same Nphi, Nr, and dealias.")

r_idx_CPC = int(np.argmin(np.abs(r_1d - r_CPC)))

### determine phi_CPC from omega_RW ###
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real
omega_RW = d3.lap(psi_RW).evaluate()
omega_RW.change_scales(dealias)

phi_idx = int(np.argmax(omega_RW['g'][:, r_idx_CPC]))
phi_CPC = 2 * np.pi * phi_idx / Nphi_deal_grid
print(f"phi_CPC = {np.degrees(phi_CPC):.2f} deg")

domega_RW_dr = (er @ d3.grad(omega_RW)).evaluate()
domega_RW_dr.change_scales(dealias)

### radial profiles ###
r_profile         = r_1d
domega_dr_profile = domega_RW_dr['g'][phi_idx, :]    # d_r(omega_RW)(r, phi_CPC)
grad_f_profile    = -gamma * r_profile                # d_r(f) = -gamma*r

### zonal profile: linear-interp across the CPC band ###
# Read band edges via linear interp on the saved zonal grid.
r_lo = max(r_profile[0],  r_CPC - rho_window)
r_hi = min(r_profile[-1], r_CPC + rho_window)
#print(r_lo, r_profile[0], r_hi, r_CPC - rho_window, r_profile[-1], r_CPC + rho_window)
omega_lo = float(np.interp(r_lo, r_1d_zonal, omega_zonal_full))
omega_hi = float(np.interp(r_hi, r_1d_zonal, omega_zonal_full))
domega_zonal_dr_at_rCPC = (omega_hi - omega_lo) / (r_hi - r_lo)

# Construct the "band-replaced" omega_zonal profile for plotting/full-r diagnostics:
omega_zonal_interp = omega_zonal_full.copy()
band_mask = (r_1d_zonal >= r_lo) & (r_1d_zonal <= r_hi)
omega_zonal_interp[band_mask] = omega_lo + (omega_hi - omega_lo) * (
    (r_1d_zonal[band_mask] - r_lo) / (r_hi - r_lo + 1e-30))

# Take d/dr of the band-replaced profile via Dedalus (so the ratio profile
# is consistent everywhere with the same differentiation convention).
omega_zonal_interp_field = dist.Field(bases=disk)
omega_zonal_interp_field.change_scales(dealias)
omega_zonal_interp_field['g'] = np.broadcast_to(
    omega_zonal_interp[np.newaxis, :], (Nphi_deal_grid, Nr_deal_grid)).copy()
domega_zonal_dr_interp_field = (er @ d3.grad(omega_zonal_interp_field)).evaluate()
domega_zonal_dr_interp_field.change_scales(dealias)
domega_zonal_dr_profile = domega_zonal_dr_interp_field['g'][0, :]

# (Sanity: domega_zonal_dr_profile at r=r_CPC should be ~ domega_zonal_dr_at_rCPC.
# They can differ slightly because Dedalus differentiation isn't a pure linear
# central-difference; we report both.)
domega_zonal_dr_at_rCPC_dedalus = float(domega_zonal_dr_profile[r_idx_CPC])

### three-term balance ###
two_term_lhs   = float(domega_dr_profile[r_idx_CPC])
three_term_lhs = two_term_lhs + domega_zonal_dr_at_rCPC
gamma_r_CPC    = gamma * r_CPC

two_term_ratio   = abs(two_term_lhs)   / (gamma_r_CPC + 1e-30)
three_term_ratio = abs(three_term_lhs) / (gamma_r_CPC + 1e-30)
# signed ratios (for direction-of-imbalance reporting)
two_term_ratio_signed   = two_term_lhs   / (gamma_r_CPC + 1e-30)
three_term_ratio_signed = three_term_lhs / (gamma_r_CPC + 1e-30)

# Full-r ratio profiles
two_term_ratio_profile   = np.abs(domega_dr_profile) / (gamma * r_profile + 1e-30)
three_term_ratio_profile = np.abs(domega_dr_profile + domega_zonal_dr_profile) \
                            / (gamma * r_profile + 1e-30)

### equilibrium radii ###
def find_zeros(diff_profile, r_profile):
    sc = np.where(np.diff(np.sign(diff_profile)))[0]
    out = []
    for s in sc:
        r_l, r_r = r_profile[s], r_profile[s+1]
        f_l, f_r = diff_profile[s], diff_profile[s+1]
        if (f_r - f_l) != 0:
            out.append(float(r_l - f_l * (r_r - r_l) / (f_r - f_l)))
    return out

r_eq_two   = find_zeros(domega_dr_profile - gamma * r_profile, r_profile)
r_eq_three = find_zeros(domega_dr_profile + domega_zonal_dr_profile
                        - gamma * r_profile, r_profile)

### print summary ###
print(f"\n{'='*70}")
print(f"Three-term force balance at r_CPC = {r_CPC:.4f}, gamma = {gamma:.0f}")
print(f"{'='*70}")
print(f"  gamma * r_CPC                = {gamma_r_CPC:+.4e}")
print(f"  d_r(omega_RW)    at r_CPC    = {two_term_lhs:+.4e}")
print(f"  d_r(omega_zonal) at r_CPC    = {domega_zonal_dr_at_rCPC:+.4e}  "
      f"(linear interp across [{r_lo:.3f}, {r_hi:.3f}])")
print(f"    (Dedalus diff of band-replaced profile: "
      f"{domega_zonal_dr_at_rCPC_dedalus:+.4e})")
print(f"  RW only          ratio (abs)  = {two_term_ratio:.4f}  "
      f"(signed: {two_term_ratio_signed:+.4f})")
print(f"  RW + zonal       ratio (abs)  = {three_term_ratio:.4f}  "
      f"(signed: {three_term_ratio_signed:+.4f})")
print(f"  Two-term  r_eq:   {r_eq_two}")
print(f"  Three-term r_eq:  {r_eq_three}")
print(f"{'='*70}")

### save results ###
results = {
    'gamma'              : gamma,
    'r_CPC'              : r_CPC,
    'phi_CPC'            : phi_CPC,
    'A_wave'             : A_wave,
    # two-term (back-compat with v6)
    'r_eq'               : np.array(r_eq_two),
    'ratio_at_rCPC'      : two_term_ratio,
    'grad_f_at_rCPC'     : float(grad_f_profile[r_idx_CPC]),
    'domega_dr_at_rCPC'  : two_term_lhs,
    # three-term additions
    'domega_zonal_dr_at_rCPC'        : domega_zonal_dr_at_rCPC,
    'domega_zonal_dr_at_rCPC_dedalus': domega_zonal_dr_at_rCPC_dedalus,
    'r_eq_three'                     : np.array(r_eq_three),
    'three_term_lhs_at_rCPC'         : three_term_lhs,
    'three_term_ratio'               : three_term_ratio,
    'three_term_ratio_signed'        : three_term_ratio_signed,
    'two_term_ratio_signed'          : two_term_ratio_signed,
    'rho_window'                     : rho_window,
    'r_band_lo'                      : r_lo,
    'r_band_hi'                      : r_hi,
    'omega_zonal_at_band_lo'         : omega_lo,
    'omega_zonal_at_band_hi'         : omega_hi,
    # metadata
    'evp_mode_idx'       : dominant_evp_idx,
    'evp_eval'           : evals_sorted[dominant_evp_idx],
    'fft_mode_idx'       : fft_mode_idx,
    'fft_eval_target'    : target_eval,
    'Nphi'               : Nphi,
    'Nr'                 : Nr,
    'n_glitch'           : n_glitch,
    'n_total'            : len(r_locs),
    'output_suffix'      : output_suffix,
    # full profiles
    'r_profile'                : r_profile,
    'domega_dr_profile'        : domega_dr_profile,
    'grad_f_profile'           : grad_f_profile,
    'domega_zonal_dr_profile'  : domega_zonal_dr_profile,
    'omega_zonal_interp'       : omega_zonal_interp,
    'omega_zonal_full'         : omega_zonal_full,
    'two_term_ratio_profile'   : two_term_ratio_profile,
    'three_term_ratio_profile' : three_term_ratio_profile,
}
np.save(save_results, results)
print(f"Results saved to: {save_results}")

### figure ###
fig, axes = plt.subplots(2, 1, figsize=(9, 9), constrained_layout=True)

# panel 1: signed gradients vs r
ax = axes[0]
ax.plot(r_profile, grad_f_profile,        color='C0', lw=1.5,
        label=r'$\partial_r f = -\gamma r$')
ax.plot(r_profile, domega_dr_profile,     color='C1', lw=1.5,
        label=r'$\partial_r \omega_\mathrm{RW}$')
ax.plot(r_profile, domega_zonal_dr_profile, color='C2', lw=1.5,
        label=r'$\partial_r \omega_\mathrm{zonal}$ (band-interp)')
ax.plot(r_profile, domega_dr_profile + domega_zonal_dr_profile,
        color='C3', lw=1.5, ls='--',
        label=r'$\partial_r \omega_\mathrm{RW} + \partial_r \omega_\mathrm{zonal}$')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
ax.axvspan(r_lo, r_hi, color='gray', alpha=0.15,
           label=f'CPC band $\\pm{rho_window:.2f}$')
for r_eq in r_eq_three:
    if 0.02 < r_eq < 0.9:
        ax.axvline(r_eq, color='green', ls='-', lw=1.0, alpha=0.7,
                   label=f'$r_{{\\rm eq, 3}}={r_eq:.4f}$')
ax.set_xlabel(r'$r$')
ax.set_ylabel('Signed radial gradient')
ax.set_title(f'Three-term force balance — $\\gamma={gamma:.0f}$, '
             f'$\\phi_{{\\rm CPC}}={np.degrees(phi_CPC):.1f}°$, $A={A_wave:.4f}$')
ax.set_xlim([0, 1])
r_hi_idx = np.where(r_profile <= r_hi)[0][-1]
yabsmax = np.max((np.abs(grad_f_profile)[:r_hi_idx], np.abs(domega_dr_profile)[:r_hi_idx], np.abs(domega_zonal_dr_profile)[:r_hi_idx], np.abs(domega_dr_profile + domega_zonal_dr_profile)[:r_hi_idx]))
ax.set_ylim(-yabsmax, yabsmax)
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# panel 2: two-term vs three-term ratio
ax = axes[1]
ax.plot(r_profile, two_term_ratio_profile,   color='C1', lw=1.2,
        label=r'$|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r$  (two-term)')
ax.plot(r_profile, three_term_ratio_profile, color='C3', lw=1.5,
        label=r'$|\partial_r\omega_\mathrm{RW} + \partial_r\omega_\mathrm{zonal}|'
              r' \,/\, \gamma r$  (three-term)')
ax.axhline(1.0, color='k', ls='--', lw=1.0, label='Balance (ratio = 1)')
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
ax.axvspan(r_lo, r_hi, color='gray', alpha=0.15)
ax.set_xlabel(r'$r$')
ax.set_ylabel('balance ratio')
ax.set_title('Force balance ratio vs radius')
ax.set_xlim([0, 1])
finite_mask = (r_profile > 0.02) & (r_profile < 0.5)
ymax = min(5, float(np.nanmax(three_term_ratio_profile[finite_mask])) * 1.2)
ax.set_ylim([0, max(1.2, ymax)])
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

fig.savefig(output, dpi=150)
plt.close(fig)
print(f"Figure saved: {output}")

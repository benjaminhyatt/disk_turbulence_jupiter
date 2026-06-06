"""
Order-of-magnitude force balance estimate for the CPC, as a function of radius.

Loads the CPC tracking file to determine r_CPC and phi_CPC automatically.
Compares the two signed radial gradients along phi = phi_CPC:

    dr(f)        = -gamma * r          (strictly negative)
    dr(omega_RW) = A * dr(nabla^2 psi_evec)(r, phi_CPC)   (signed)

Balance condition: dr(omega_RW)(r, phi_CPC) = gamma * r

Saves a results dictionary for use in a downstream summary plot.

Usage:
    process_force_balance_oom.py [options]

Options:
    --evp_file=<str>        path to processed EVP .npy file
    --fft_file=<str>        path to processed FFT projection .npy file
    --tracking_file=<str>   path to processed tracking .npy file
    --gamma=<float>         gamma value [default: 675.0]
    --Nphi=<int>            azimuthal resolution [default: 512]
    --Nr=<int>              radial resolution [default: 256]
    --output=<str>          output figure filename [default: force_balance_oom.png]
    --save_results=<str>    path to save results dictionary [default: force_balance_oom_results.npy]
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

evp_file      = args['--evp_file']
fft_file      = args['--fft_file']
tracking_file = args['--tracking_file']
gamma         = float(args['--gamma'])
Nphi          = int(args['--Nphi'])
Nr            = int(args['--Nr'])
output        = args['--output']
save_results  = args['--save_results']
dealias       = 3/2

### load tracking file and compute r_CPC, phi_CPC ###
logger.info("Loading tracking file: " + tracking_file)
tracking = np.load(tracking_file, allow_pickle=True)[()]

r_locs   = np.array(tracking['r_locs'],   dtype=float)
phi_locs = np.array(tracking['phi_locs'], dtype=float)

# glitch mask: exclude points where r drops below half the mean
r_mean_raw   = np.mean(r_locs)
glitch_mask  = r_locs < 0.5 * r_mean_raw
n_glitch     = np.sum(glitch_mask)
r_clean      = r_locs[~glitch_mask]
phi_clean    = phi_locs[~glitch_mask]

# r_CPC: mean of clean radial positions
r_CPC = np.mean(r_clean)

### WRONG: phi_CPC: mean azimuthal angle using circular mean to handle wrapping
#phi_CPC_circ = np.arctan2(np.mean(np.sin(phi_clean)), np.mean(np.cos(phi_clean)))
#if phi_CPC_circ < 0:
#    phi_CPC_circ += 2 * np.pi
#phi_CPC = phi_CPC_circ

print(f"\nTracking file: {tracking_file}")
print(f"  Total points: {len(r_locs)},  glitches excluded: {n_glitch}")
print(f"  r_CPC (mean, clean)   = {r_CPC:.4f}")
#print(f"  phi_CPC (circ. mean)  = {np.degrees(phi_CPC):.2f} deg")


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

#phi_idx   = np.argmin(np.abs(phi_1d - phi_CPC))
r_idx_CPC = np.argmin(np.abs(r_1d   - r_CPC))
### load EVP file ###
logger.info("Loading EVP file")
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

dominant_evp_idx = 0
psi_evec_real    = evecs_sorted[dominant_evp_idx].real
print(f"\nDominant EVP mode (idx={dominant_evp_idx}): "
      f"eval={evals_sorted[dominant_evp_idx].real:.4f}+i{evals_sorted[dominant_evp_idx].imag:.4f}")

### determine phase
Nphi_deal = psi_evec_real.shape[0]
phi_idx = np.argmax(-psi_evec_real[:, r_idx_CPC])
phi_CPC = 2 * np.pi * phi_idx / Nphi_deal
print(f"  phi_CPC_evp  = {np.degrees(phi_CPC):.2f} deg")


print(f"\nGrid point closest to (r_CPC={r_CPC:.4f}, phi_CPC={np.degrees(phi_CPC):.2f} deg):")
print(f"  r_grid   = {r_1d[r_idx_CPC]:.4f}  (idx={r_idx_CPC})")
print(f"  phi_grid = {np.degrees(phi_1d[phi_idx]):.2f} deg  (idx={phi_idx})")

## load FFT file ###
logger.info("Loading FFT file")
fft_data = np.load(fft_file, allow_pickle=True)[()]
A_wave   = fft_data['projdot_amp_extrema'][0]
print(f"Projection amplitude (extrema): A = {A_wave:.6f}")

### build fields ###
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real

omega_RW     = d3.lap(psi_RW).evaluate()
omega_RW.change_scales(dealias)

# check phase
Nphi_deal = omega_RW['g'].shape[0]
phi_idx = np.argmax(omega_RW['g'][:, r_idx_CPC])
phi_CPC = 2 * np.pi * phi_idx / Nphi_deal
print(f"  phi_CPC_evp  = {np.degrees(phi_CPC):.2f} deg")

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

r_idx_max  = np.argmax(np.abs(domega_dr_profile))
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
    'A_wave'             : A_wave,
    'r_eq'               : np.array(r_eq_vals),
    'r_max_grad'         : r_max_grad,
    'max_grad'           : max_grad,
    'ratio_at_rCPC'      : float(ratio_profile[r_idx_CPC]),
    'grad_f_at_rCPC'     : float(grad_f_profile[r_idx_CPC]),
    'domega_dr_at_rCPC'  : float(domega_dr_profile[r_idx_CPC]),
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

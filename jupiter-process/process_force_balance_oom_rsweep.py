"""
Order-of-magnitude force balance estimate for the CPC, as a function of radius.

Compares the two radial gradients along the fixed azimuthal slice phi = phi_CPC:

    dr(f)        = -gamma * r          (inward, magnitude = gamma * r)
    dr(omega_RW) = A * dr(nabla^2 psi_evec)(r, phi_CPC)

Balance condition: gamma * r = A * dr(omega_RW)(r, phi_CPC)
i.e. ratio(r) = |dr(omega_RW)| / (gamma * r) = 1

Outputs:
  - Printed table of both gradients and their ratio vs r
  - Figure showing both gradients and their ratio vs r, with:
      * r_CPC marked (observed CPC orbital radius from tracking)
      * equilibrium radius r_eq where ratio = 1
      * radius of maximum dr(omega_RW)

Usage:
    process_force_balance_oom.py [options]

Options:
    --evp_file=<str>        path to processed EVP .npy file
    --fft_file=<str>        path to processed FFT projection .npy file
    --gamma=<float>         gamma value [default: 675.0]
    --r_CPC=<float>         mean CPC orbital radius from tracking [default: 0.1558]
    --phi_CPC_deg=<float>   CPC azimuthal angle in degrees [default: 96.09]
    --Nphi=<int>            azimuthal resolution [default: 512]
    --Nr=<int>              radial resolution [default: 256]
    --output=<str>          output figure filename [default: force_balance_oom.png]
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

evp_file    = args['--evp_file']
fft_file    = args['--fft_file']
gamma       = float(args['--gamma'])
r_CPC       = float(args['--r_CPC'])
phi_CPC     = np.radians(float(args['--phi_CPC_deg']))
Nphi        = int(args['--Nphi'])
Nr          = int(args['--Nr'])
output      = args['--output']
dealias     = 3/2

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

# azimuthal index closest to phi_CPC
phi_idx = np.argmin(np.abs(phi_1d - phi_CPC))
print(f"\nUsing phi slice: phi_grid={np.degrees(phi_1d[phi_idx]):.2f} deg (idx={phi_idx})")

# radial index closest to r_CPC
r_idx_CPC = np.argmin(np.abs(r_1d - r_CPC))
print(f"r_CPC grid point: r={r_1d[r_idx_CPC]:.4f} (idx={r_idx_CPC})")

### load EVP file ###
logger.info("Loading EVP file")
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

dominant_evp_idx = 1
psi_evec_real    = evecs_sorted[dominant_evp_idx].real
print(f"\nDominant EVP mode (idx={dominant_evp_idx}): "
      f"eval={evals_sorted[dominant_evp_idx].real:.4f}+i{evals_sorted[dominant_evp_idx].imag:.4f}")

### load FFT file ###
logger.info("Loading FFT file")
fft_data = np.load(fft_file, allow_pickle=True)[()]
A_wave   = fft_data['projdot_amp_extrema'][0]
print(f"Projection amplitude (extrema): A = {A_wave:.6f}")

### build fields ###
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real

omega_RW = d3.lap(psi_RW).evaluate()
omega_RW.change_scales(dealias)

domega_RW_dr = (er @ d3.grad(omega_RW)).evaluate()
domega_RW_dr.change_scales(dealias)

### extract radial profiles along phi_CPC slice ###
# full radial grid, all points
r_profile          = r_1d                                        # shape (Nr_deal,)
domega_dr_profile  = domega_RW_dr['g'][phi_idx, :]              # dr(omega_RW) vs r
omega_RW_profile   = omega_RW['g'][phi_idx, :]                  # omega_RW vs r
grad_f_profile     = gamma * r_profile                           # |dr(f)| = gamma*r
ratio_profile      = np.abs(domega_dr_profile) / (grad_f_profile + 1e-30)

### find equilibrium radius where ratio = 1 ###
# look for sign change in (|dr(omega_RW)| - gamma*r) — use the signed version
signed_diff = domega_dr_profile - grad_f_profile   # note: domega_dr can be negative

# find where ratio crosses 1 (i.e. |domega_dr| = gamma*r)
ratio_minus_one = ratio_profile - 1.0
sign_changes = np.where(np.diff(np.sign(ratio_minus_one)))[0]

r_eq_vals = []
for sc in sign_changes:
    # linear interpolation between sc and sc+1
    r_lo, r_hi = r_profile[sc], r_profile[sc+1]
    f_lo, f_hi = ratio_minus_one[sc], ratio_minus_one[sc+1]
    r_eq = r_lo - f_lo * (r_hi - r_lo) / (f_hi - f_lo)
    r_eq_vals.append(r_eq)

# radius of maximum |dr(omega_RW)|
r_idx_max   = np.argmax(np.abs(domega_dr_profile))
r_max_grad  = r_profile[r_idx_max]
max_grad_val = np.abs(domega_dr_profile[r_idx_max])

### print results ###
print(f"\n{'='*65}")
print(f"Radial force balance scan along phi={np.degrees(phi_1d[phi_idx]):.1f} deg")
print(f"{'='*65}")
print(f"{'r':>8}  {'dr(f)':>12}  {'dr(omRW)':>12}  {'ratio':>8}")
print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
# print every few points to keep output manageable
step = max(1, len(r_profile) // 30)
for ri in range(0, len(r_profile), step):
    marker = ''
    if abs(r_profile[ri] - r_CPC) < 2*(r_profile[1]-r_profile[0]):
        marker = '  <-- r_CPC'
    if abs(r_profile[ri] - r_max_grad) < 2*(r_profile[1]-r_profile[0]):
        marker = '  <-- max dr(omega_RW)'
    print(f"{r_profile[ri]:>8.4f}  {grad_f_profile[ri]:>12.4e}  "
          f"{domega_dr_profile[ri]:>12.4e}  {ratio_profile[ri]:>8.4f}{marker}")

print(f"\nEquilibrium radius/radii where ratio=1:")
if len(r_eq_vals) == 0:
    print("  None found in domain")
for r_eq in r_eq_vals:
    print(f"  r_eq = {r_eq:.4f}  (cf. r_CPC = {r_CPC:.4f})")

print(f"\nRadius of maximum |dr(omega_RW)|: r = {r_max_grad:.4f}")
print(f"  max |dr(omega_RW)| = {max_grad_val:.4e}")
print(f"  gamma * r_max      = {gamma * r_max_grad:.4e}")
print(f"  ratio at r_max     = {max_grad_val / (gamma * r_max_grad):.4f}")
print(f"\nAt r_CPC = {r_CPC:.4f}:")
print(f"  |dr(f)|      = {gamma * r_CPC:.4e}")
print(f"  |dr(omRW)|   = {np.abs(domega_dr_profile[r_idx_CPC]):.4e}")
print(f"  ratio        = {ratio_profile[r_idx_CPC]:.4f}")
print(f"{'='*65}")

### figure ###
fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)

# panel 1: both gradients vs r
ax = axes[0]
ax.plot(r_profile, grad_f_profile,            color='C0', lw=1.5,
        label=r'$|\partial_r f| = \gamma r$')
ax.plot(r_profile, np.abs(domega_dr_profile), color='C1', lw=1.5,
        label=r'$|\partial_r \omega_\mathrm{RW}|$')
ax.plot(r_profile, domega_dr_profile,         color='C1', lw=0.8,
        ls='--', alpha=0.5, label=r'$\partial_r \omega_\mathrm{RW}$ (signed)')
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
ax.axvline(r_max_grad, color='C1', ls=':', lw=1.0,
           label=f'max $|\\partial_r\\omega_{{\\rm RW}}|$ at $r={r_max_grad:.4f}$')
for r_eq in r_eq_vals:
    ax.axvline(r_eq, color='green', ls='-', lw=1.2,
               label=f'$r_{{\\rm eq}}={r_eq:.4f}$')
ax.set_xlabel(r'$r$')
ax.set_ylabel('Radial gradient magnitude')
ax.set_title(f'Force balance scan — $\\gamma={gamma:.0f}$, '
             f'$\\phi_{{\\rm CPC}}={np.degrees(phi_CPC):.1f}°$, $A={A_wave:.4f}$')
ax.legend(fontsize=8)
ax.set_xlim([0, 1])
ax.grid(True, alpha=0.3)

# panel 2: ratio vs r
ax = axes[1]
ax.plot(r_profile, ratio_profile, color='C3', lw=1.5,
        label=r'$|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r$')
ax.axhline(1.0, color='k', ls='--', lw=1.0, label='Balance (ratio=1)')
ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
           label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
for r_eq in r_eq_vals:
    ax.axvline(r_eq, color='green', ls='-', lw=1.2,
               label=f'$r_{{\\rm eq}}={r_eq:.4f}$')
ax.axvline(r_max_grad, color='C1', ls=':', lw=1.0,
           label=f'max $|\\partial_r\\omega_{{\\rm RW}}|$ at $r={r_max_grad:.4f}$')
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'Ratio $|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r$')
ax.set_title('Force balance ratio vs radius')
ax.set_xlim([0, 1])
ax.set_ylim([0, min(5, ratio_profile[r_profile > 0.01].max() * 1.2)])
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.savefig(output, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {output}")

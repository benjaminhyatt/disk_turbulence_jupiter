"""
Force balance calculation for a circumpolar cyclone (CPC) in the gamma-plane.

Models the CPC as a smooth Gaussian vorticity disk and computes two radial
force terms:

1. F_gamma: inward gamma-drift force from the Coriolis parameter gradient
       F_gamma = integ(omega_CPC * (-gamma * r))
   where -gamma*r = d(f)/dr is the radial gradient of the Coriolis parameter
   f(r) = -0.5*gamma*r^2.

2. F_RW: outward force from the radial gradient of the Rossby wave streamfunction
       F_RW = integ(omega_CPC * er . grad(psi_RW))
   where psi_RW = A * psi_evec (dominant EVP mode, scaled by projection amplitude)
   and er is the radial unit vector.

The CPC is placed at the vorticity crest of the dominant Rossby wave
(streamfunction trough), at the mean tracking radius r_CPC.

The force balance condition F_gamma + F_RW = 0 determines the equilibrium
radius. Here we evaluate both forces at r_CPC for a parameter sweep over
CPC radius a and peak vorticity omega_0.

Sign convention: nabla^2(psi) = omega (no minus sign).
The Rossby wave vorticity crest = streamfunction trough.
The CPC (positive vorticity) sits at the vorticity crest of the wave.

Usage:
    python force_balance.py
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### parameters ###
gamma   = 675.0
r_CPC   = 0.1558       # mean CPC orbital radius from tracking
phi_CPC = np.radians(96.09)  # vorticity crest azimuth from eigenmode analysis

# projection amplitude of dominant mode (from FFT fitting, row 0 after amp sort)
# using the projdot amplitude from the mpm processed file
A_wave  = 0.1316       # dominant mode amplitude (from projdot_fit_params[0,0])

# CPC parameter sweep
a_vals      = np.array([0.10, 0.15, 0.20, 0.25, 0.30])   # CPC radius
omega0_vals = np.arange(20, 220, 20, dtype=float)          # CPC peak vorticity

# EVP and projection files
evp_file = (
    "../jupiter-process/"
    "processed_rossby_evp_m_1_inviscid_0_u0bg_0_"
    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_"
    "tau_mod_1_seed_10001_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

### Dedalus setup ###
m_evp   = 1
Nphi    = 512
Nr      = 256
dealias = 3/2

dtype   = np.float64
coords  = d3.PolarCoordinates('phi', 'r')
dist    = d3.Distributor(coords, dtype=dtype)
disk    = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                       dealias=dealias, dtype=dtype)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))

# radial unit vector field
er = dist.VectorField(coords, bases=disk)
er.change_scales(dealias)
er['g'][1] = 1.0   # er = (0, 1) in (phi, r) components

### load dominant EVP mode and scale by projection amplitude ###
logger.info("Loading EVP file")
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']   # stored as psi grid data (complex)
sort_idxs    = np.argsort(evals.imag)
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

dominant_evp_idx = 1
psi_evec_complex = evecs_sorted[dominant_evp_idx]   # complex, shape (Nphi_deal, Nr_deal)

print(f"Dominant mode: eval={evals_sorted[dominant_evp_idx].real:.4f}+"
      f"i{evals_sorted[dominant_evp_idx].imag:.4f}")
print(f"Projection amplitude A_wave = {A_wave:.4f}")

# the eigenmode grid data is complex; we need the real part
# (the physical wave is the real part of the complex eigenmode)
# we also need to orient the eigenmode so that the streamfunction trough
# (vorticity crest) is at phi_CPC
# the trough of Re[psi_evec] is already at ~96 deg from our earlier analysis
psi_evec_real = psi_evec_complex.real   # shape (Nphi_deal, Nr_deal)

# build the Rossby wave psi field: psi_RW = A_wave * psi_evec_real
# note: psi_evec_real already has the correct azimuthal orientation
# (vorticity crest at phi ~ 96 deg)
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real

print(f"psi_RW range: [{np.min(psi_RW['g']):.4e}, {np.max(psi_RW['g']):.4e}]")

### Coriolis parameter gradient field ###
# f(r) = -0.5*gamma*r^2  =>  df/dr = -gamma*r
# this is the coefficient that enters the gamma-drift force integrand
df_dr = dist.Field(bases=disk)
df_dr.change_scales(dealias)
df_dr['g'] = -gamma * r_g

### precompute radial gradient of psi_RW ###
# d(psi_RW)/dr evaluated on the grid
dpsi_RW_dr = (er @ d3.grad(psi_RW)).evaluate()
dpsi_RW_dr.change_scales(dealias)

print(f"d(psi_RW)/dr at r~r_CPC, phi~phi_CPC: approx "
      f"{dpsi_RW_dr['g'][Nphi*3//4//2, Nr//4]:.4e}")

### CPC vorticity field (Gaussian disk) ###
# Gaussian centered at (phi_CPC, r_CPC) in Cartesian distance
x_CPC = r_CPC * np.cos(phi_CPC)
y_CPC = r_CPC * np.sin(phi_CPC)
x_g   = r_g * np.cos(phi_g)
y_g   = r_g * np.sin(phi_g)
dist2 = (x_g - x_CPC)**2 + (y_g - y_CPC)**2   # squared Cartesian distance

omega_CPC = dist.Field(bases=disk)

### force balance sweep ###
# results arrays
F_gamma_arr = np.zeros((len(a_vals), len(omega0_vals)))
F_RW_arr    = np.zeros((len(a_vals), len(omega0_vals)))
Gamma_arr   = np.zeros((len(a_vals), len(omega0_vals)))

logger.info("Starting parameter sweep")
for i, a in enumerate(a_vals):
    # Gaussian sigma: we set sigma = a/2 so that the field is ~14% of peak at r=a
    sigma = a / 5.0

    for j, omega0 in enumerate(omega0_vals):
        # set CPC vorticity field
        omega_CPC.change_scales(dealias)
        omega_CPC['g'] = omega0 * np.exp(-dist2 / (2 * sigma**2))

        # total circulation
        Gamma = d3.integ(omega_CPC).evaluate()['g'][0, 0]
        Gamma_arr[i, j] = Gamma

        # F_gamma = integ(omega_CPC * df_dr)
        # df_dr = -gamma*r, so this gives inward (negative r) force tendency
        F_gamma_field = (omega_CPC * df_dr).evaluate()
        F_gamma = d3.integ(F_gamma_field).evaluate()['g'][0, 0]
        F_gamma_arr[i, j] = F_gamma

        # F_RW = integ(omega_CPC * d(psi_RW)/dr)
        F_RW_field = (omega_CPC * dpsi_RW_dr).evaluate()
        F_RW = d3.integ(F_RW_field).evaluate()['g'][0, 0]
        F_RW_arr[i, j] = F_RW

    logger.info(f"a={a:.2f} done")

### print summary table ###
print(f"\n{'a':>6}  {'omega0':>8}  {'Gamma':>10}  {'F_gamma':>12}  {'F_RW':>12}  "
      f"{'F_RW/F_gamma':>14}  {'ratio sign':>10}")
for i, a in enumerate(a_vals):
    for j, omega0 in enumerate(omega0_vals):
        ratio = F_RW_arr[i,j] / (F_gamma_arr[i,j] + 1e-30)
        print(f"{a:>6.2f}  {omega0:>8.1f}  {Gamma_arr[i,j]:>10.4f}  "
              f"{F_gamma_arr[i,j]:>12.4e}  {F_RW_arr[i,j]:>12.4e}  "
              f"{ratio:>14.4f}  "
              f"{'BALANCE' if abs(ratio + 1) < 0.2 else ''}")

### figure ###
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# panel 1: F_gamma vs omega0 for each a (should be linear in omega0)
ax = axes[0]
for i, a in enumerate(a_vals):
    ax.plot(omega0_vals, F_gamma_arr[i,:], marker='o', ms=4,
            label=f'a={a:.2f}')
ax.set_xlabel('omega_0 (peak CPC vorticity)')
ax.set_ylabel('F_gamma')
ax.set_title('Inward gamma-drift force')
ax.legend(fontsize=8)
ax.axhline(0, color='k', lw=0.5)

# panel 2: F_RW vs omega0 for each a
ax = axes[1]
for i, a in enumerate(a_vals):
    ax.plot(omega0_vals, F_RW_arr[i,:], marker='o', ms=4,
            label=f'a={a:.2f}')
ax.set_xlabel('omega_0 (peak CPC vorticity)')
ax.set_ylabel('F_RW')
ax.set_title('Outward Rossby wave force')
ax.legend(fontsize=8)
ax.axhline(0, color='k', lw=0.5)

# panel 3: F_RW / |F_gamma| — ratio of outward to inward force
# balance occurs when this ratio = -1
ax = axes[2]
for i, a in enumerate(a_vals):
    ratio = F_RW_arr[i,:] / (np.abs(F_gamma_arr[i,:]) + 1e-30)
    ax.plot(omega0_vals, ratio, marker='o', ms=4, label=f'a={a:.2f}')
ax.axhline(-1, color='k', ls='--', lw=1.0, label='balance (ratio=-1)')
ax.axhline(0,  color='k', lw=0.5)
ax.set_xlabel('omega_0 (peak CPC vorticity)')
ax.set_ylabel('F_RW / |F_gamma|')
ax.set_title('Force ratio (balance at -1)')
ax.legend(fontsize=8)

fig.suptitle(
    f'Force balance sweep — gamma={gamma}, r_CPC={r_CPC:.4f}, A_wave={A_wave:.4f}',
    fontsize=11
)
fig.savefig('force_balance_sweep.png', dpi=150)
plt.close(fig)
print("\nFigure saved: force_balance_sweep.png")

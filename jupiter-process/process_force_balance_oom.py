"""
Order-of-magnitude force balance estimate for the CPC.

Compares the two radial gradients at r = r_CPC:

    |dr(f)|          = gamma * r_CPC
    |dr(omega_RW)|   = A * |dr(nabla^2(psi_evec))| evaluated at (r_CPC, phi_CPC)

Balance requires: gamma * r_CPC ~ A * dr(omega_RW_evec)(r_CPC, phi_CPC)

The ratio dr(omega_RW) / (gamma * r_CPC) tells us how close to balance we are.
A ratio of 1.0 indicates exact balance; << 1 means the wave is too weak.

A = projdot_amp_extrema[0] from the FFT fitting output (dominant mode,
amplitude-sorted index 0), which is the coefficient such that
psi_RW = A * psi_evec in the biorthogonal projection.

The vorticity gradient is evaluated at (r_CPC, phi_CPC) where the CPC sits
at the wave vorticity crest (phi_CPC ~ 96 deg for the gamma=675 example).

Usage:
    force_balance_oom.py [options]

Options:
    --evp_file=<str>        path to processed EVP .npy file
    --fft_file=<str>        path to processed FFT projection .npy file
    --gamma=<float>         gamma value [default: 675.0]
    --r_CPC=<float>         mean CPC orbital radius from tracking [default: 0.1558]
    --phi_CPC_deg=<float>   CPC azimuthal angle in degrees [default: 96.09]
    --Nphi=<int>            azimuthal resolution [default: 512]
    --Nr=<int>              radial resolution [default: 256]
"""

import numpy as np
import dedalus.public as d3
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
dealias     = 3/2

### Dedalus setup ###
coords       = d3.PolarCoordinates('phi', 'r')
dist         = d3.Distributor(coords, dtype=np.float64)
disk         = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                            dealias=dealias, dtype=np.float64)
phi_g, r_g   = dist.local_grids(disk, scales=(dealias, dealias))

er = dist.VectorField(coords, bases=disk)
er.change_scales(dealias)
er['g'][1] = 1.0

### find grid indices closest to (r_CPC, phi_CPC) ###
r_1d   = r_g[0, :]
phi_1d = phi_g[:, 0]
r_idx   = np.argmin(np.abs(r_1d   - r_CPC))
phi_idx = np.argmin(np.abs(phi_1d - phi_CPC))

print(f"\nGrid point closest to (r_CPC={r_CPC:.4f}, phi_CPC={np.degrees(phi_CPC):.2f} deg):")
print(f"  r_grid={r_1d[r_idx]:.4f} (idx={r_idx})")
print(f"  phi_grid={np.degrees(phi_1d[phi_idx]):.2f} deg (idx={phi_idx})")

### load EVP file and extract dominant mode ###
logger.info("Loading EVP file: " + evp_file)
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)        # sort_im_inc
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

dominant_evp_idx = 1   # confirmed: eval ~ 13.71+i0.019 for gamma=675
psi_evec_real    = evecs_sorted[dominant_evp_idx].real

print(f"\nDominant EVP mode (idx={dominant_evp_idx}):")
print(f"  eval = {evals_sorted[dominant_evp_idx].real:.4f} + i{evals_sorted[dominant_evp_idx].imag:.4f}")

### load FFT fitting file and extract amplitude ###
logger.info("Loading FFT file: " + fft_file)
fft_data    = np.load(fft_file, allow_pickle=True)[()]
A_wave      = fft_data['projdot_amp_extrema'][0]   # dominant mode, amplitude-sorted index 0

print(f"\nProjection amplitude (extrema estimate): A = {A_wave:.6f}")

### build Rossby wave vorticity field and its radial gradient ###
psi_RW = dist.Field(bases=disk)
psi_RW.change_scales(dealias)
psi_RW['g'] = A_wave * psi_evec_real

# omega_RW = nabla^2(psi_RW)  [sign convention: nabla^2(psi) = omega]
omega_RW = d3.lap(psi_RW).evaluate()
omega_RW.change_scales(dealias)

# radial gradient of omega_RW
domega_RW_dr = (er @ d3.grad(omega_RW)).evaluate()
domega_RW_dr.change_scales(dealias)

### evaluate at CPC location ###
domega_dr_at_CPC = domega_RW_dr['g'][phi_idx, r_idx]
omega_RW_at_CPC  = omega_RW['g'][phi_idx, r_idx]

### compute the two gradient magnitudes ###
grad_f_mag      = gamma * r_CPC                  # |dr(f)| = gamma * r_CPC
grad_omegaRW_mag = np.abs(domega_dr_at_CPC)      # |dr(omega_RW)|

ratio = grad_omegaRW_mag / grad_f_mag

### print results ###
print(f"\n{'='*60}")
print(f"Order-of-magnitude force balance estimate")
print(f"{'='*60}")
print(f"gamma                        = {gamma:.1f}")
print(f"r_CPC                        = {r_CPC:.4f}")
print(f"phi_CPC                      = {np.degrees(phi_CPC):.2f} deg")
print(f"A_wave (extrema amplitude)   = {A_wave:.6f}")
print(f"")
print(f"|dr(f)| = gamma * r_CPC      = {grad_f_mag:.6e}")
print(f"|dr(omega_RW)| at CPC        = {grad_omegaRW_mag:.6e}")
print(f"  omega_RW at CPC            = {omega_RW_at_CPC:.6e}")
print(f"  dr(omega_RW) at CPC        = {domega_dr_at_CPC:.6e}")
print(f"")
print(f"Ratio |dr(omega_RW)| / |dr(f)| = {ratio:.4f}")
print(f"  (balance requires ratio ~ 1.0)")
if ratio > 0.5 and ratio < 2.0:
    print(f"  --> ORDER OF MAGNITUDE BALANCE ACHIEVED")
elif ratio > 0.1:
    print(f"  --> Within an order of magnitude, but not balanced")
else:
    print(f"  --> Wave too weak to balance gamma-drift by factor ~{1/ratio:.1f}")
print(f"{'='*60}")

### also show the unnormalized eigenmode gradient for reference ###
psi_evec_field = dist.Field(bases=disk)
psi_evec_field.change_scales(dealias)
psi_evec_field['g'] = psi_evec_real

omega_evec     = d3.lap(psi_evec_field).evaluate()
omega_evec.change_scales(dealias)
domega_evec_dr = (er @ d3.grad(omega_evec)).evaluate()
domega_evec_dr.change_scales(dealias)

print(f"\nFor reference (unnormalized eigenmode):")
print(f"  dr(nabla^2 psi_evec) at CPC = {domega_evec_dr['g'][phi_idx, r_idx]:.6e}")
print(f"  A * dr(nabla^2 psi_evec)    = {A_wave * domega_evec_dr['g'][phi_idx, r_idx]:.6e}")
print(f"  (should match dr(omega_RW) above)")

"""
Compute the phase offset between the dominant Rossby wave vorticity crest
and the CPC position, for gamma=675.

Mathematical setup (derived from first principles):
---------------------------------------------------
1. Real Fourier basis convention in Dedalus:
   The IVP streamfunction is expanded as:
       psi(r,phi) = sum_m [ A_m(r)*cos(m*phi) + B_m(r)*(-sin(m*phi)) ]
   so projdot_c is the coefficient of cos(m*phi) and
      projdot_s is the coefficient of -sin(m*phi).

2. The wave's lab-frame streamfunction phase angle is:
       phi_psi(t) = atan2(-projdot_s, projdot_c)
   with a minus sign on projdot_s to correct for the -sin convention.

3. The eigenmode's cosine and sine radial profiles have norms norm_A and
   norm_B respectively (generally unequal due to the arbitrary complex
   phase of the EVP eigenvector). This causes (projdot_c, projdot_s) to
   trace an ellipse rather than a circle in phase space, distorting the
   atan2 result. The fix is to normalize before taking atan2:
       phi_psi(t) = atan2(-projdot_s/norm_B, projdot_c/norm_A)
   This recovers the correct uniformly-evolving lab-frame angle.

4. Sign convention: nabla^2(psi) = omega (no minus sign).
   With Laplacian eigenvalue -k^2: omega = -k^2 * psi.
   So the vorticity CREST = streamfunction TROUGH, i.e. the two are
   exactly pi radians apart in azimuth. Converting from streamfunction
   phase to vorticity crest phase:
       phi_wave(t) = (phi_psi(t) + pi) % (2*pi)

5. The CPC tracker finds vorticity maxima, so phi_CPC is the vorticity
   crest of the CPC — directly comparable to phi_wave.

6. Phase offset:
       delta_phi(t) = angle( exp(i*phi_wave) * exp(-i*phi_CPC) )
   gives the signed angular separation in (-pi, pi], handling wrapping.
   A mean near 0 means the CPC sits at the wave vorticity crest.

Usage:
    process_evec_phase.py <evp_file> <fft_file> <tracking_file>
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### file paths ###
#evp_file = (
#    "../jupiter-process/"
#    "processed_rossby_evp_m_1_inviscid_0_u0bg_0_"
#    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
#    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_"
#    "tau_mod_1_seed_10001_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
#)

#projection_file = (
#    "../jupiter-process/"
#    "processed_rossby_projection_mpm_sort_im_inc_m_1_inviscid_0_u0bg_0_"
#    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_"
#    "alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_10001_"
#    "safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
#)

#tracking_file = (
#    "../jupiter-process/"
#    "processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
#    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_1_tau_mod_1_"
#    "seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
#)

from docopt import docopt
args = docopt(__doc__)
print(args)

evp_file = args['<evp_file>']
projection_file = args['<fft_file>']
tracking_file = args['<tracking_file>']

### parameters ###
m       = 1
Nphi    = 512
Nr      = 256
dealias = 3/2
r_CPC   = 0.156

### set up Dedalus grid ###
coords       = d3.PolarCoordinates('phi', 'r')
dist         = d3.Distributor(coords, dtype=np.complex128)
disk         = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                            dealias=dealias, dtype=np.complex128)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
r_1d       = r_deal[0, :]
phi_1d     = phi_deal[:, 0]
Nphi_deal  = len(phi_1d)

r_peak_idx = np.argmin(np.abs(r_1d - r_CPC))
print(f"Closest radial grid point to r_CPC={r_CPC}: r={r_1d[r_peak_idx]:.4f} (idx={r_peak_idx})")

### load and sort EVP file ###
logger.info("Loading EVP file")
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evals.imag)        # sort_im_inc
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

print(f"\nFirst 5 eigenvalues (sort_im_inc):")
for k in range(5):
    print(f"  idx={k}: {evals_sorted[k].real:.4f}+i{evals_sorted[k].imag:.4f}")

### dominant mode ###
dominant_evp_idx = 1   # confirmed: eval ~ 13.71+i0.019
psi_dom = evecs_sorted[dominant_evp_idx]
print(f"\nDominant mode EVP idx={dominant_evp_idx}: "
      f"eval={evals_sorted[dominant_evp_idx].real:.4f}+i{evals_sorted[dominant_evp_idx].imag:.4f}")

### compute norm_A and norm_B from EVP coefficients ###
# set up complex distributor to access spectral coefficients
dtype_co     = np.complex128
dist_co      = d3.Distributor(d3.PolarCoordinates('phi', 'r'), dtype=dtype_co)
disk_co      = d3.DiskBasis(d3.PolarCoordinates('phi', 'r'),
                            shape=(Nphi, Nr), radius=1,
                            dealias=dealias, dtype=dtype_co)

# m_map helper (same as projection script)
def m_map(m, Nphi, flag):
    m_in = np.array([m])
    if flag == 'co':
        m_out = 2 * m_in
        mask = m_in < 0
        m_out[mask] += Nphi + 1
        return m_out
    raise ValueError("Invalid flag")

# rebuild distributor properly for coefficient access
coords_co    = d3.PolarCoordinates('phi', 'r')
dist_co      = d3.Distributor(coords_co, dtype=np.complex128)
disk_co      = d3.DiskBasis(coords_co, shape=(Nphi, Nr), radius=1,
                            dealias=dealias, dtype=np.complex128)
psi_evec_field = dist_co.Field(bases=disk_co)
psi_evec_field.change_scales(dealias)
psi_evec_field['g'] = np.copy(psi_dom)

m_idx_co_plus  = int(2 * m)
m_idx_co_minus = int(Nphi + 1 - 2 * m)

c_plus  = psi_evec_field['c'][m_idx_co_plus,  :]
c_minus = psi_evec_field['c'][m_idx_co_minus, :]

A_r = c_plus.real + c_minus.real   # cosine radial profile
B_r = c_plus.imag - c_minus.imag   # sine radial profile (for -sin term)

norm_A = np.sqrt(np.sum(A_r**2))
norm_B = np.sqrt(np.sum(B_r**2))

print(f"\nEigenmode radial profile norms:")
print(f"  norm_A (cosine) = {norm_A:.4f}")
print(f"  norm_B (sine)   = {norm_B:.4f}")
print(f"  ratio norm_B/norm_A = {norm_B/norm_A:.4f}")

Nphi_deal    = psi_dom.shape[0]
phi_peak_idx = np.argmax(-psi_dom.real[:, r_peak_idx])
phi_evec     = 2 * np.pi * phi_peak_idx / Nphi_deal
print(f"phi_evec = {phi_evec:.4f} rad  ({np.degrees(phi_evec):.2f} deg)")

### load projection and tracking ###
proj     = np.load(projection_file, allow_pickle=True)[()]
tracking = np.load(tracking_file,   allow_pickle=True)[()]

tw     = proj['tw']
tw_tr  = tracking['tw']

if not np.allclose(tw, tw_tr, rtol=1e-6):
    print("\nWARNING: time axes do not match")
else:
    print(f"\nTime axes agree: {len(tw)} points, t in [{tw[0]:.3f}, {tw[-1]:.3f}]")

# dominant mode after amplitude sorting is row 0
projdot_c = proj['projdot_c'][0, :]
projdot_s = proj['projdot_s'][0, :]
phi_cpc   = np.array(tracking['phi_locs'], dtype=float)
r_cpc_t   = np.array(tracking['r_locs'],   dtype=float)

### glitch mask ###
r_mean      = np.mean(r_cpc_t)
glitch_mask = r_cpc_t < 0.5 * r_mean
n_glitch    = np.sum(glitch_mask)
print(f"\nGlitch points: {n_glitch} / {len(tw)} ({100*n_glitch/len(tw):.1f}%)")

### compute phi_wave: lab-frame vorticity crest angle of wave ###
# step 1: normalize projdot components to correct for elliptical distortion
# step 2: atan2 with -projdot_s/norm_B (minus sign from -sin convention)
# step 3: add pi to convert from streamfunction crest to vorticity crest
#         (because omega = nabla^2(psi) = -k^2*psi, so omega crest = psi trough)
phi_psi  = np.arctan2(-projdot_s / norm_B, projdot_c / norm_A)
phi_wave = (phi_psi + np.pi) % (2 * np.pi)

### phase offset: wave vorticity crest vs CPC vorticity maximum ###
delta_phi = np.angle(np.exp(1j * phi_wave) * np.exp(-1j * phi_cpc))

# statistics excluding glitches
delta_clean = delta_phi[~glitch_mask]
print(f"\nPhase offset phi_wave - phi_CPC (glitches excluded):")
print(f"  mean   = {np.degrees(np.mean(delta_clean)):.2f} deg")
print(f"  median = {np.degrees(np.median(delta_clean)):.2f} deg")
print(f"  std    = {np.degrees(np.std(delta_clean)):.2f} deg")
print(f"\nInterpretation:")
print(f"  ~0 deg   -> CPC sits at wave vorticity crest")
print(f"  ~90 deg  -> wave vorticity crest leads CPC by 90 deg")
print(f"  ~-90 deg -> CPC leads wave vorticity crest by 90 deg")

### figure ###
fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

# panel 1: (projdot_c/norm_A, -projdot_s/norm_B) phase portrait
# should trace a circle if normalization is correct
ax = axes[0]
ax.plot(projdot_c / norm_A, -projdot_s / norm_B, color='C0', lw=0.5, alpha=0.7)
ax.set_aspect('equal')
ax.set_xlabel('projdot_c / norm_A')
ax.set_ylabel('-projdot_s / norm_B')
ax.set_title('Phase portrait after normalization — should be circular')
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.4, label='unit circle')
ax.legend(fontsize=8)

# panel 2: instantaneous phase offset over time
ax = axes[1]
ax.plot(tw[~glitch_mask], np.degrees(delta_phi[~glitch_mask]),
        color='C3', lw=0.8, label='phase offset (clean)')
if n_glitch > 0:
    ax.scatter(tw[glitch_mask], np.degrees(delta_phi[glitch_mask]),
               color='red', s=10, zorder=5, label='glitch points')
ax.axhline(np.degrees(np.mean(delta_clean)), color='k', ls='--', lw=1.0,
           label=f'mean = {np.degrees(np.mean(delta_clean)):.1f} deg')
ax.axhline(np.degrees(np.mean(delta_clean)) + np.degrees(np.std(delta_clean)),
           color='gray', ls=':', lw=0.8)
ax.axhline(np.degrees(np.mean(delta_clean)) - np.degrees(np.std(delta_clean)),
           color='gray', ls=':', lw=0.8,
           label=f'±1sigma = {np.degrees(np.std(delta_clean)):.1f} deg')
ax.set_xlabel('time')
ax.set_ylabel('phase offset (deg)')
ax.set_title('phi_wave_vorticity - phi_CPC  (corrected)')
ax.legend(fontsize=8)

# panel 3: phi_wave and phi_CPC overlaid
ax = axes[2]
ax.plot(tw, np.degrees(phi_wave),              color='C0', lw=1.0,
        label='phi_wave (wave vorticity crest)')
ax.plot(tw, np.degrees(phi_cpc % (2*np.pi)),   color='C1', lw=1.0, alpha=0.8,
        label='phi_CPC (tracker)')
if n_glitch > 0:
    ax.scatter(tw[glitch_mask], np.degrees(phi_cpc[glitch_mask] % (2*np.pi)),
               color='red', s=10, zorder=5, label='glitch points')
ax.set_xlabel('time')
ax.set_ylabel('azimuthal angle (deg)')
ax.set_title('Wave vorticity crest vs. CPC position in lab frame')
ax.legend(fontsize=8)

fig.suptitle(r'Phase relationship: wave vorticity crest vs. CPC  ($\gamma=675$)',
             fontsize=11)
fig.savefig('evec_phase_comparison.png', dpi=150)
plt.close(fig)
print(f"\nFigure saved: evec_phase_comparison.png")

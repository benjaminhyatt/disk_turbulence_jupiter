"""
Compute the eigenmode vorticity crest phase phi_evec for the dominant m=1
Rossby mode, and compare with the CPC tracking angle to determine the
physical phase offset between the CPC and the wave vorticity crest.

Sign convention: nabla^2(psi) = omega (no minus sign).
Consequently omega = -k^2 * psi, so the vorticity crest corresponds to
the streamfunction TROUGH (minimum of psi), not the maximum.

phi_evec is defined as the azimuthal angle of the vorticity crest of the
eigenmode at r = r_CPC, i.e. the argmin of psi_dom.real at that radius.

The physical wave vorticity crest angle in the lab frame is:
    phi_wave(t) = atan2(projdot_s(t), projdot_c(t)) - phi_evec

Note: projdot tracks the streamfunction projection, so the phase of
projdot_s/projdot_c also follows the streamfunction convention. The
vorticity crest is offset by pi from the streamfunction crest, which
is already accounted for by using argmin(psi) = argmax(omega).

The CPC tracker finds maxima of vorticity, so phi_CPC corresponds to
the vorticity crest of the CPC, which we compare directly to phi_wave.

Usage:
    python compute_evec_phase.py
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### file paths ###
evp_file = (
    "../jupiter-process/"
    "processed_rossby_evp_m_1_inviscid_0_u0bg_0_"
    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_"
    "tau_mod_1_seed_10001_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

projection_file = (
    "../jupiter-process/"
    "processed_rossby_projection_mpm_sort_im_inc_m_1_inviscid_0_u0bg_0_"
    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_"
    "alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_10001_"
    "safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

tracking_file = (
    "../jupiter-process/"
    "processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_1_tau_mod_1_"
    "seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

### parameters ###
m       = 1
Nphi    = 512
Nr      = 256
dealias = 3/2
r_CPC   = 0.156   # mean CPC orbital radius

### set up Dedalus grid to find r index ###
coords       = d3.PolarCoordinates('phi', 'r')
dist         = d3.Distributor(coords, dtype=np.complex128)
disk         = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                            dealias=dealias, dtype=np.complex128)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

r_1d       = r_deal[0, :]
r_peak_idx = np.argmin(np.abs(r_1d - r_CPC))
Nphi_deal  = phi_deal.shape[0]
phi_1d     = phi_deal[:, 0]

print(f"Closest radial grid point to r_CPC={r_CPC}: r={r_1d[r_peak_idx]:.4f} (idx={r_peak_idx})")

### load EVP file and sort by Im(eval) increasing ###
logger.info("Loading EVP file")
evp          = np.load(evp_file, allow_pickle=True)[()]
evals        = evp['evals_res']
evecs_psi    = evp['psi_right_evecs_res']   # stored as psi grid data
sort_idxs    = np.argsort(evals.imag)       # sort_im_inc
evals_sorted = evals[sort_idxs]
evecs_sorted = evecs_psi[sort_idxs]

print(f"\nTotal resolved EVP modes: {len(evals_sorted)}")
print(f"First 5 eigenvalues (sort_im_inc):")
for k in range(5):
    print(f"  idx={k}: {evals_sorted[k].real:.4f}+i{evals_sorted[k].imag:.4f}")

### dominant mode: EVP sort index 1 ###
# (confirmed from amplitude sort order [1 0 2 3...] in projection fitting)
dominant_evp_idx = 1
psi_dom = evecs_sorted[dominant_evp_idx]
print(f"\nDominant mode EVP idx={dominant_evp_idx}: "
      f"eval={evals_sorted[dominant_evp_idx].real:.4f}+i{evals_sorted[dominant_evp_idx].imag:.4f}")
print(f"psi_dom.real range: [{np.min(psi_dom.real):.4f}, {np.max(psi_dom.real):.4f}]")
print(f"psi_dom.imag range: [{np.min(psi_dom.imag):.4f}, {np.max(psi_dom.imag):.4f}]")

### compute phi_evec: azimuthal angle of vorticity crest ###
# sign convention: omega = nabla^2(psi) = -k^2 * psi
# so vorticity crest = streamfunction trough = argmin(psi.real)
# we use only Re[psi] since the eigenmode is real-valued physically
psi_at_r     = psi_dom.real[:, r_peak_idx]
phi_peak_idx = np.argmin(psi_at_r)   # vorticity crest = psi trough
phi_evec     = phi_1d[phi_peak_idx]

print(f"\nphi_evec (vorticity crest of eigenmode at r={r_1d[r_peak_idx]:.4f}):")
print(f"  phi_peak_idx = {phi_peak_idx}")
print(f"  phi_evec = {phi_evec:.4f} rad  ({np.degrees(phi_evec):.2f} deg)")

### load projection and tracking data ###
proj    = np.load(projection_file, allow_pickle=True)[()]
tracking = np.load(tracking_file,  allow_pickle=True)[()]

print(proj['evals_re'])

tw      = proj['tw']
tw_tr   = tracking['tw']

if not np.allclose(tw, tw_tr, rtol=1e-6):
    print("\nWARNING: time axes do not match")
else:
    print(f"\nTime axes agree: {len(tw)} points, t in [{tw[0]:.3f}, {tw[-1]:.3f}]")

# dominant mode projdot_c and projdot_s (after amplitude sorting, row 0)
projdot_c = proj['projdot_c'][0, :]
projdot_s = proj['projdot_s'][0, :]

# CPC azimuthal angle (vorticity maximum tracker)
phi_cpc = np.array(tracking['phi_locs'], dtype=float)

### compute physical phase offset ###
# raw projection phase (in streamfunction convention)
#phi_proj_raw = np.arctan2(-projdot_s, projdot_c)
norm_A = 0.1069  # ||A(r)||, cosine profile norm
norm_B = 0.9943  # ||B(r)||, sine profile norm
phi_proj_raw = np.arctan2(projdot_s / norm_B, projdot_c / norm_A)

# correct for eigenmode phase to get vorticity crest angle in lab frame
#phi_wave = (phi_proj_raw + phi_evec) % (2 * np.pi)
#phi_wave = phi_proj_raw % (2 * np.pi)
phi_wave = (phi_proj_raw + np.pi) % (2 * np.pi)

# phase offset: wave vorticity crest vs CPC vorticity maximum
# use complex exponential to handle wrapping cleanly
# note: CPC orbits in negative phi direction, consistent with exp(-i*phi_cpc)
# from earlier analysis

delta_phi = np.angle(np.exp(1j * phi_wave) * np.exp(-1j * phi_cpc))

print(f"\nPhase offset: phi_wave_vorticity - phi_CPC")
print(f"  mean   = {np.degrees(np.mean(delta_phi)):.2f} deg")
print(f"  median = {np.degrees(np.median(delta_phi)):.2f} deg")
print(f"  std    = {np.degrees(np.std(delta_phi)):.2f} deg")
print(f"\nInterpretation:")
print(f"  ~0 deg  -> CPC sits at wave vorticity crest")
print(f"  ~90 deg -> CPC leads wave vorticity crest by 90 deg")

### figure ###
fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

# panel 1: psi_dom.real at r_CPC vs phi
ax = axes[0]
ax.plot(np.degrees(phi_1d), psi_at_r, color='C0', lw=1.2)
ax.axvline(np.degrees(phi_evec), color='C3', ls='--', lw=1.2,
           label=f'vorticity crest (psi trough): phi_evec={np.degrees(phi_evec):.1f} deg')
ax.axvline(np.degrees(phi_evec) + 180, color='C2', ls=':', lw=1.0,
           label=f'psi crest (vorticity trough): {np.degrees(phi_evec)+180:.1f} deg')
ax.set_xlabel('phi (deg)')
ax.set_ylabel('Re[psi_evec]')
ax.set_title(f'Dominant eigenmode streamfunction at r={r_1d[r_peak_idx]:.4f}  '
             f'(nabla^2 psi = omega, so omega crest = psi trough)')
ax.legend(fontsize=9)

# panel 2: instantaneous phase offset over time
ax = axes[1]
ax.plot(tw, np.degrees(delta_phi), color='C3', lw=0.8)
ax.axhline(np.degrees(np.mean(delta_phi)), color='k', ls='--', lw=1.0,
           label=f'mean = {np.degrees(np.mean(delta_phi)):.1f} deg')
ax.axhline(np.degrees(np.mean(delta_phi)) + np.degrees(np.std(delta_phi)),
           color='gray', ls=':', lw=0.8)
ax.axhline(np.degrees(np.mean(delta_phi)) - np.degrees(np.std(delta_phi)),
           color='gray', ls=':', lw=0.8, label=f'±1sigma = {np.degrees(np.std(delta_phi)):.1f} deg')
ax.set_xlabel('time')
ax.set_ylabel('phase offset (deg)')
ax.set_title('phi_wave_vorticity - phi_CPC  (corrected for eigenmode phase)')
ax.legend(fontsize=9)

# panel 3: phi_wave and phi_CPC overlaid (normalised to [0,1])
ax = axes[2]
phi_wave_wrapped = phi_wave % (2 * np.pi)
phi_cpc_wrapped  = phi_cpc  % (2 * np.pi)
ax.plot(tw, np.degrees(phi_wave_wrapped), color='C0', lw=1.0,
        label='phi_wave (vorticity crest, from projection)')
ax.plot(tw, np.degrees(phi_cpc_wrapped),  color='C1', lw=1.0, alpha=0.8,
        label='phi_CPC (from tracker)')
ax.set_xlabel('time')
ax.set_ylabel('azimuthal angle (deg)')
ax.set_title('Wave vorticity crest vs. CPC position in lab frame')
ax.legend(fontsize=9)

fig.suptitle(r'Phase relationship: wave vorticity crest vs. CPC  ($\gamma=675$)',
             fontsize=11)
fig.savefig('evec_phase_comparison.png', dpi=150)
plt.close(fig)
print(f"\nFigure saved: evec_phase_comparison.png")

"""
Compute the eigenmode phase angle phi_evec for the dominant m=1 Rossby mode
from the processed EVP output, for gamma=675.

The eigenmode streamfunction has the form:
    psi_evec(r, phi) = 2*A(r)*cos(m*phi) - 2*B(r)*sin(m*phi)

where A(r) = Re[psi_hat_+(r)] and B(r) = Im[psi_hat_+(r)] are the cosine
and sine radial profiles respectively (the _c_re and _s_re quantities from
the projection script).

The eigenmode phase angle is defined as:
    phi_evec = atan2(||2B(r)||_2, ||2A(r)||_2)

where ||.||_2 is the L2 norm over r. This is the angle in the (cos, sin)
plane that characterizes the eigenmode's azimuthal orientation.

The physically meaningful wave phase in the lab frame is then:
    phi_wave(t) = atan2(projdot_s(t), projdot_c(t)) - phi_evec

Usage:
    python compute_evec_phase.py
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### parameters ###
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

m       = 1
Nphi    = 512
Nr      = 256
dealias = 3/2

# sorting applied in the projection script (to identify which EVP mode is i=1)
sort_1  = 'im'
sort_2  = 'inc'

# after amplitude-sorting in the MPM script, the dominant mode is index 0
# but we need to know which EVP index it corresponds to
# load the projection file to get the amplitude sort order
proj = np.load(projection_file, allow_pickle=True)[()]

### helper: same m_map as in projection script ###
def m_map(m, Nphi, flag):
    m_in = np.array(m)
    if not m_in.shape:
        m_in = np.array([m])
    if flag == 're':
        m_out = 4 * m_in
        mask = m_out > Nphi - 2
        m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
        return m_out
    elif flag == 'co':
        m_out = 2 * m_in
        mask = m_in < 0
        m_out[mask] += Nphi + 1
        return m_out
    else:
        raise ValueError("Invalid flag: " + flag)

m_idx_co_plus  = m_map(m,  Nphi, 'co')[0]
m_idx_co_minus = m_map(-m, Nphi, 'co')[0]

### load EVP file ###
logger.info("Loading EVP file: " + evp_file)
evp = np.load(evp_file, allow_pickle=True)[()]

evals      = evp['evals_res']
drifts     = evp['drifts_res']
evecs_psi  = evp['psi_right_evecs_res']   # shape (n_modes, Nphi_deal, Nr_deal)

# apply same sorting logic as projection script
sort_by  = evals.imag
sort_idxs = np.argsort(sort_by)   # 'im', 'inc'
evals_sorted     = evals[sort_idxs]
evecs_psi_sorted = evecs_psi[sort_idxs]

print(f"Total resolved EVP modes: {len(evals_sorted)}")
print(f"First 5 eigenvalues (sorted): {evals_sorted[:5]}")

### set up Dedalus basis ###
dtype_co = np.complex128
coords   = d3.PolarCoordinates('phi', 'r')
dist_co  = d3.Distributor(coords, dtype=dtype_co)
disk_co  = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                        dealias=dealias, dtype=dtype_co)

psi_evec = dist_co.Field(bases=disk_co)

### loop over first few modes and compute phi_evec ###
n_modes_check = min(5, len(evals_sorted))

print(f"\nComputing phi_evec for first {n_modes_check} EVP modes (sorted by Im, inc):")
print(f"{'idx':>4}  {'eval':>20}  {'||A||':>10}  {'||B||':>10}  {'phi_evec (deg)':>15}")

phi_evec_all = []
for idx in range(n_modes_check):
    psi_evec.change_scales(dealias)
    psi_evec['g'] = np.copy(evecs_psi_sorted[idx])

    # extract cosine and sine radial profiles from complex coefficients
    # following exactly the same convention as process_rossby_projections_v2.py
    c_plus  = psi_evec['c'][m_idx_co_plus,  :]
    c_minus = psi_evec['c'][m_idx_co_minus, :]

    A_r = c_plus.real + c_minus.real   # 2*A(r): cosine radial profile
    B_r = c_plus.imag - c_minus.imag   # 2*B(r): sine radial profile

    norm_A = np.sqrt(np.sum(A_r**2))
    norm_B = np.sqrt(np.sum(B_r**2))

    phi_evec = np.arctan2(norm_B, norm_A)
    phi_evec_all.append(phi_evec)

    print(f"{idx:>4}  {evals_sorted[idx].real:>10.4f}+i{evals_sorted[idx].imag:>8.4f}  "
          f"{norm_A:>10.4f}  {norm_B:>10.4f}  {np.degrees(phi_evec):>15.2f}")

# the dominant mode after amplitude sorting corresponds to EVP index 1
# (from the amplitude sort order [1 0 2 3 ...] seen in the fitting output)
# i.e., the physically dominant mode is EVP sort index 1
dominant_evp_idx = 1
phi_evec_dominant = phi_evec_all[dominant_evp_idx]

print(f"\nDominant mode (EVP sort idx={dominant_evp_idx}):")
print(f"  phi_evec = {phi_evec_dominant:.4f} rad  ({np.degrees(phi_evec_dominant):.2f} deg)")
print(f"\nPhysical wave phase = atan2(projdot_s, projdot_c) - phi_evec")
print(f"  i.e., subtract {np.degrees(phi_evec_dominant):.2f} deg from the atan2 time series")
print(f"  to get the wave crest azimuthal angle in the lab frame.")

### quick check: load projection and compute mean physical phase offset vs CPC ###
# load tracking
tracking_file = (
    "../jupiter-process/"
    "processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_1_tau_mod_1_"
    "seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)
tracking = np.load(tracking_file, allow_pickle=True)[()]

phi_cpc   = np.array(tracking['phi_locs'], dtype=float)
t_track   = tracking['tw']

# dominant mode projdot_c and projdot_s
# after amplitude sorting, dominant mode is row 0
projdot_c_dom = proj['projdot_c'][0, :]
projdot_s_dom = proj['projdot_s'][0, :]
t_proj        = proj['tw']

# check time axes agree
if np.allclose(t_track, t_proj, rtol=1e-6):
    print(f"\nTime axes agree: {len(t_track)} points")

    # raw atan2 phase of projection
    phi_proj_raw = np.arctan2(projdot_s_dom, projdot_c_dom)

    # corrected wave crest phase
    phi_wave = phi_proj_raw - phi_evec_dominant

    # phase offset: CPC vs wave crest
    # use complex exponential difference to handle wrapping
    delta_phi = np.angle(np.exp(-1j * phi_cpc) * np.exp(1j * phi_wave))

    print(f"\nPhase offset phi_wave - phi_CPC (corrected for eigenmode phase):")
    print(f"  mean   = {np.degrees(np.mean(delta_phi)):.2f} deg")
    print(f"  median = {np.degrees(np.median(delta_phi)):.2f} deg")
    print(f"  std    = {np.degrees(np.std(delta_phi)):.2f} deg")
    print(f"\nInterpretation:")
    print(f"  0 deg   -> CPC sits at wave crest")
    print(f"  90 deg  -> CPC leads wave crest by 90 deg")
    print(f"  -90 deg -> CPC trails wave crest by 90 deg")
else:
    print("\nWARNING: time axes do not match — skipping phase offset calculation")

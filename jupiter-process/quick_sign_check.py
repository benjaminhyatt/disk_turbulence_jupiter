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


evp = np.load(evp_file, allow_pickle=True)[()]
evecs_psi = evp['psi_right_evecs_res']
sort_idxs = np.argsort(evp['evals_res'].imag)
evecs_psi_sorted = evecs_psi[sort_idxs]

# dominant mode grid data, shape (Nphi_deal, Nr_deal)
psi_dom = evecs_psi_sorted[1]
print("shape:", psi_dom.shape)
print("dtype:", psi_dom.dtype)
print("max real:", np.max(psi_dom.real))
print("max imag:", np.max(psi_dom.imag))
print("sample real (mid-phi, mid-r):", psi_dom.real[psi_dom.shape[0]//4, psi_dom.shape[1]//2])
print("sample imag (mid-phi, mid-r):", psi_dom.imag[psi_dom.shape[0]//4, psi_dom.shape[1]//2])


# find phi index of maximum of Re[psi] at the radial location of peak amplitude
psi_dom = evecs_psi_sorted[1]
# radial index of peak (averaged over phi)
r_peak_idx = np.argmax(np.mean(np.abs(psi_dom.real), axis=0))
# azimuthal index of maximum at that radius
phi_peak_idx = np.argmax(psi_dom.real[:, r_peak_idx])
#print(np.where(np.max(psi_dom.real[:, r_peak_idx]) - psi_dom.real[:, r_peak_idx] == 0))
#print(np.where(np.max(np.max(psi_dom.real[:, r_peak_idx]) - psi_dom.real[:, r_peak_idx]) == np.max(psi_dom.real[:, r_peak_idx]) - psi_dom.real[:, r_peak_idx]))

# convert to angle
Nphi_deal = psi_dom.shape[0]
phi_evec = 2 * np.pi * phi_peak_idx / Nphi_deal
print(f"phi_evec = {phi_evec:.4f} rad  ({np.degrees(phi_evec):.2f} deg)")










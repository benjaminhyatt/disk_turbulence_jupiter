import dedalus.public as d3
import numpy as np

Nphi, Nr = 512, 256
dealias  = 3/2
coords   = d3.PolarCoordinates('phi', 'r')
dist     = d3.Distributor(coords, dtype=np.complex128)
disk     = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                        dealias=dealias, dtype=np.complex128)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

r_1d      = r_deal[0, :]   # radial grid, shape (Nr_deal,)
r_cpc     = 0.156
r_peak_idx = np.argmin(np.abs(r_1d - r_cpc))
print(f"r_1d[r_peak_idx] = {r_1d[r_peak_idx]:.4f}")

evp = np.load('../jupiter-process/processed_rossby_evp_m_1_inviscid_0_u0bg_0_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_10001_safety_1d0em01_timestepper_SBDF2_bc_sf.npy', allow_pickle=True)[()]
evecs_psi    = evp['psi_right_evecs_res']
sort_idxs    = np.argsort(evp['evals_res'].imag)
evecs_sorted = evecs_psi[sort_idxs]
psi_dom      = evecs_sorted[1]

Nphi_deal    = psi_dom.shape[0]
phi_peak_idx = np.argmax(-psi_dom.real[:, r_peak_idx])
phi_evec     = 2 * np.pi * phi_peak_idx / Nphi_deal
print(f"phi_evec = {phi_evec:.4f} rad  ({np.degrees(phi_evec):.2f} deg)")

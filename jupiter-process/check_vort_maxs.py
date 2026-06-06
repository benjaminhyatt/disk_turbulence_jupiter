import numpy as np

processed_tracking = np.load('processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_10001_safety_1d0em01_timestepper_SBDF2_bc_sf.npy', allow_pickle=True)[()]

print(np.mean(processed_tracking['vort_maxs']))

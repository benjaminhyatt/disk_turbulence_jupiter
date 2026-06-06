import numpy as np
import h5py
import scipy.special as sp
import dedalus.public as d3

### Setup 
def std_roots(m, Nr):
    return sp.jn_zeros(m, Nr)

def std_weights(m, std_zsm):
    return (sp.jv(m + 1, std_zsm)**2) / 2

processed = np.load('processed_spectra_zb_mbin_std_steady_nu_2em04_gam_9d5ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy', allow_pickle=True)[()]

Nphi = 512
Nr = 256

dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

u = dist.VectorField(coords, name = 'u', bases = disk) # velocity
vort = dist.Field(name = 'vort', bases = disk) # scalar vertical vorticity

tw = processed['ts']
t_steady_range=218
tavg_end = tw[-1]
tavg_start = tavg_end - t_steady_range
tavg_end_idx = np.where(tw <= tavg_end)[0][-1]
tavg_start_idx = np.where(tw >= tavg_start)[0][0]

m = 0
std_zs_0 = std_roots(m, Nr)

#psiB = processed['psiB']
#psiB0 = psiB[:, 0, :]
#keBmn = processed['keBmn']
#keB0n = keBmn[:, 0, :]

keBmn_tavg = processed['keBmn_tavg']
keB0n_tavg = keBmn_tavg[0, :]

# calculating more...
#psiB_tavg = np.mean(psiB[tavg_start_idx:tavg_end_idx, :, :], axis = 0) # wasn't saved...
#psiB_tavg = processed['psiB_tavg'] # will have...
psiB0n_tavg = np.sqrt( keB0n_tavg / ( np.pi * (std_zs_0)**2 * std_weights(0, std_zs_0) ) )

# consistency check...
#print("psiB_tavg: m=0,", psiB_tavg[0, :])
print("psiB0n_tavg, from keB0n,", psiB0n_tavg)

quick_save = np.array(psiB0n_tavg)
np.save('quick_save.npy', quick_save)

print("attempting imports")
import numpy as np
import matplotlib.pyplot as plt
import h5py
import dedalus.public as d3
print("past imports")

#Nphi, Nr = 256, 128
Nphi, Nr = 512, 256
dealias = 3/2 
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)

nu = 2e-4
alpha = 1e-2

j0 = 0
j = -1

# load in analysis
print("attempting to load")
f = h5py.File('../jupiter-run/analysis_debug_512_256_long/analysis_debug_512_256_long_s1.h5')
print("attempting to index")
t = f['tasks/W'].dims[0]['sim_time'][j0:j]

# scalars
W = f['tasks/W'][j0:j, 0, 0]
W *= 1/(2*np.pi)

print("attemping to load")
f = h5py.File('../jupiter-run/analysis_debug_aux_512_256_long/analysis_debug_aux_512_256_long_s1.h5')
print("attempting to index")
W_wz_lap_rate = f['tasks/lap_rate'][j0:j, 0, 0]
W_wz_tau_rate = f['tasks/tau_rate'][j0:j, 0, 0]
W_wz_tau_rate_2 = f['tasks/tau_rate_2'][j0:j, 0, 0]
W_wz_adv_rate = f['tasks/adv_rate'][j0:j, 0, 0]
W_wz_for_rate = f['tasks/for_rate'][j0:j, 0, 0]
W_wz_dam_rate = f['tasks/dam_rate'][j0:j, 0, 0]
W_wz_lap_rate *= 1/(2*np.pi)
W_wz_tau_rate *= 1/(2*np.pi)
W_wz_tau_rate_2 *= 1/(2*np.pi)
W_wz_adv_rate *= 1/(2*np.pi)
W_wz_for_rate *= 1/(2*np.pi)
W_wz_dam_rate *= 1/(2*np.pi)

W_wz = f['tasks/wz'][j0:j, 0, 0]
W_wz_tau = f['tasks/W_tau'][j0:j, 0, 0]
W_wz_tau_2 = f['tasks/W_tau_2'][j0:j, 0, 0]
W_wz_lap = f['tasks/W_lap'][j0:j, 0, 0]
W_wz_for = f['tasks/W_for'][j0:j, 0, 0]
W_wz_adv = f['tasks/W_adv'][j0:j, 0, 0]
W_wz_dam = f['tasks/W_dam'][j0:j, 0, 0]
W_wz_lap *= 1/(2*np.pi)
W_wz_tau *= 1/(2*np.pi)
W_wz_tau_2 *= 1/(2*np.pi)
W_wz_adv *= 1/(2*np.pi)
W_wz_for *= 1/(2*np.pi)
W_wz_dam *= 1/(2*np.pi)

# save processed data
processed = {}

processed['t'] = t

# instantaneous rates from laplacian term 
processed['lap_rate'] = W_wz_lap_rate

# instantaneous rates from tau term(s)
processed['tau_rate'] = W_wz_tau_rate
processed['tau_rate_2'] = W_wz_tau_rate_2

# instantaneous rates from advection term
processed['adv_rate'] = W_wz_adv_rate

# instantaneous rates from forcing term
processed['for_rate'] = W_wz_for_rate

# time-integrated data
processed['W_lap'] = W_wz_lap
processed['W_tau'] = W_wz_tau
processed['W_tau_2'] = W_wz_tau_2
processed['W_adv'] = W_wz_adv
processed['W_for'] = W_wz_for
processed['W_dam'] = W_wz_dam
processed['W_sum'] = W_wz_lap + W_wz_tau + W_wz_tau_2 + W_wz_adv + W_wz_for + W_wz_dam
processed['W'] = W

print('saving output')
np.save('processed_debug_512_256_long_no_pred.npy',processed)

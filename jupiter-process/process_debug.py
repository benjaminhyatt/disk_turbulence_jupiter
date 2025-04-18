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
tau_u = f['tasks/tau_u'][j0:j, :, :, 0] # tau_u_c
u_wz_c = f['tasks/u_wz_c'][j0:j, :, :, :]
F_c = f['tasks/F_c'][j0:j, :, :, :]
lap_u_c = f['tasks/lap_u_c'][j0:j, :, :, :]
u_c = f['tasks/u_c'][j0:j, :, :, :]

amp = -1
F_c *= amp

# scalars
W = f['tasks/W'][j0:j, 0, 0]
W *= 1/(2*np.pi)

drw0_1 = f['tasks/drw0_1'][j0:j, 0, 0]
F_1 = f['tasks/F_1'][j0:j, 0, 0]

# profiles
w0 = f['tasks/w0'][j0:j, 0, :]
drw0 = f['tasks/drw0'][j0:j, 0, :]
ur0 = f['tasks/ur0'][j0:j, 0, :]
ur0_drw0 = f['tasks/ur0_drw0'][j0:j, 0, :]

# process scalars
drw0_1 = nu*drw0_1
#tau_term = (2*Nr / np.sqrt(2*Nr-1)) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
#sig_Nr = - (2*Nr / np.sqrt(2*Nr-1)) * (np.sqrt(2*Nr-3) / (2*Nr-2))
#tau_term_2 = sig_Nr * ((2*Nr-2) / np.sqrt(2*Nr-3)) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
tau_term = np.sqrt(2*Nr) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
sig_Nr = -np.sqrt(Nr)/np.sqrt(Nr-1)
tau_term_2 = sig_Nr * np.sqrt(2*(Nr-1)) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
aW_term = alpha*W

ur0_field = dist.Field(name='ur0', bases=radial_basis)
drw0_field = dist.Field(name='drw0', bases=radial_basis)
prods = np.zeros(t.shape[0])
for i in range(t.shape[0]):
    ur0_field.change_scales(1)
    drw0_field.change_scales(1)
    ur0_field['g'] = ur0[i, :]
    drw0_field['g'] = drw0[i, :]
    prod = ur0_field * drw0_field
    prods[i] = (1/(2*np.pi))*d3.integ(prod)['g'][0, 0]

ns = np.arange(0, Nr)
dr_integral = 2*(ns + 1)/np.sqrt(2*ns + 1) # also absorbs sqrt(2)/2 coefficient
#dr3_integral = 2*ns*(ns + 1)*np.sqrt(2*ns + 1)*((2/3)*ns**4 + (4/3)*ns**3 - (5/3)*ns**2 - (7/3)*ns + 1) # also absorbs sqrt(2)/2 coefficient
dr3_integral = 2*ns*(ns + 1)*np.sqrt(2*ns + 1)*(ns**2 + ns - 1)

u_wz_cos0_n_minus = u_wz_c[:, 0, 0, :]
u_wz_cos0_n_plus = u_wz_c[:, 1, 0, :]
F_sin0_n_minus = F_c[:, 0, 1, :]
F_sin0_n_plus = F_c[:, 1, 1, :]
lap_u_sin0_n_minus = lap_u_c[:, 0, 1, :]
lap_u_sin0_n_plus = lap_u_c[:, 1, 1, :]
# use u coeffs here
u_sin0_n_minus = u_c[:, 0, 1, :]
u_sin0_n_plus = u_c[:, 1, 1, :]

adv_term = -np.sum((u_wz_cos0_n_minus + u_wz_cos0_n_plus)*dr_integral, axis = 1)
for_term = -np.sum((F_sin0_n_minus - F_sin0_n_plus)*dr_integral, axis = 1)
lap_term = -nu*np.sum((lap_u_sin0_n_minus - lap_u_sin0_n_plus)*dr_integral, axis = 1)
# new lap_term here
#lap_term_new = -nu*np.sum(((u_sin0_n_minus - u_sin0_n_plus)*dr3_integral)[:,:Nr-2], axis = 1)

lap_term_new = -nu*np.sum(((u_sin0_n_minus - u_sin0_n_plus)*dr3_integral)[:, :Nr-1], axis = 1)
lap_term_new = -nu*np.sum((u_sin0_n_minus - u_sin0_n_plus)*dr3_integral, axis = 1)

f = h5py.File('../jupiter-run/analysis_debug_aux_512_256_long/analysis_debug_aux_512_256_long_s1.h5')
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
processed['lap_rate_pred_old'] = drw0_1
processed['lap_rate_pred'] = lap_term
processed['lap_rate_pred_new'] = lap_term_new
processed['lap_rate'] = W_wz_lap_rate

# instantaneous rates from tau term(s)
processed['tau_rate_pred'] = tau_term
processed['tau_rate'] = W_wz_tau_rate
processed['tau_rate_pred_2'] = tau_term_2
processed['tau_rate_2'] = W_wz_tau_rate_2

# instantaneous rates from advection term
processed['adv_rate_pred'] = adv_term
processed['adv_rate'] = W_wz_adv_rate

# instantaneous rates from forcing term
processed['for_rate_pred'] = for_term ## not currently looking right
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

processed['w0_tavg'] = np.mean(w0, axis = 0)
processed['drw0_tavg'] = np.mean(drw0, axis = 0)

print('saving output')
np.save('processed_debug_512_256_long.npy',processed)

import numpy as np
import h5py

Nphi, Nr = 512, 256#1024, 512 
nu = 2e-3 #5e-5 #2e-4
gamma = 240 #30
k_force = 20 #50


#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.0e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.0e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
output_suffix = output_suffix.replace('-','m').replace('+','p')

# load in analysis
print("loading primary run")
f1 = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')

t = f1['tasks/W'].dims[0]['sim_time'][:]

W = f1['tasks/W'][:, 0, 0]
W *= 1/(2*np.pi)

Lzu = f1['tasks/Lzu'][:, 0, 0]
Lzu *= 1/(2*np.pi)

FW = f1['tasks/FW'][:, 0, 0]
FW *= 1/(2*np.pi)

LzF = f1['tasks/LzF'][:, 0, 0]
LzF *= 1/(2*np.pi)

KE = f1['tasks/KE'][:, 0, 0]
KE *= 1/(2*np.pi)

EN = f1['tasks/EN'][:, 0, 0]
EN *= 1/(2*np.pi)

processed = {}
processed['t'] = t
processed['W'] = W
processed['Lzu'] = Lzu
processed['FW'] = FW
processed['LzF'] = LzF
processed['KE'] = KE
processed['EN'] = EN

print('saving output')
np.save('processed_scalars_' + output_suffix + '.npy', processed)

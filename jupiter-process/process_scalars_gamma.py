import numpy as np
import h5py

gammai = 4e2
gammaf = 2.4e2
dgamma = -4e1
#ke_dt = 1e1 
#ke_rtol_mm = 2e-1
#ke_rtol_lin = 1e-1

seed_in = 31415926 #10101 #31415926 

tau_mod = True
Nphi, Nr = 512, 256 
nu = 2e-4
k_force = 20
alpha = 1e-2
amp = 1
ring = 0
eps = amp**2

output_suffix = 'nu_{:.0e}'.format(nu)
output_suffix += '_gami_{:.1e}'.format(gammai)
output_suffix += '_gamf_{:.1e}'.format(gammaf)
output_suffix += '_dgam_{:.1e}'.format(dgamma)
output_suffix += '_kf_{:.1e}'.format(k_force)
output_suffix += '_Nphi_{:}'.format(Nphi)
output_suffix += '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_seed_{:d}'.format(seed_in)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')



# load in analysis
print("loading primary run")
f1 = h5py.File('../jupiter-process/vm_analysis_' + output_suffix + '.h5')
#t = f1['tasks/KE'].dims[0]
#print(t.keys())
t = f1['tasks/KE'].dims[0]['sim_time'][:]

W = f1['tasks/W'][:, 0, 0]
Lzu = f1['tasks/Lzu'][:, 0, 0]
KE = f1['tasks/KE'][:, 0, 0]
EN = f1['tasks/EN'][:, 0, 0]

ENbdry = f1['tasks/ENbdry'][:, 0, 0]
PA = f1['tasks/PA'][:, 0, 0]
PAbdry1 = f1['tasks/PAbdry1'][:, 0, 0]
PAbdry2 = f1['tasks/PAbdry2'][:, 0, 0]


# temporal averages
tdur = 10 
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

W_tavg = np.mean(W[tstartidx:tendidx])
Lzu_tavg = np.mean(Lzu[tstartidx:tendidx])
KE_tavg = np.mean(KE[tstartidx:tendidx])
EN_tavg = np.mean(EN[tstartidx:tendidx])
KE_tavg_expected = ((eps / np.pi) - nu * EN_tavg) / (2 * alpha)

ENbdry_tavg = np.mean(ENbdry[tstartidx:tendidx])
PA_tavg = np.mean(PA[tstartidx:tendidx])
PAbdry1_tavg = np.mean(PAbdry1[tstartidx:tendidx])
PAbdry2_tavg = np.mean(PAbdry2[tstartidx:tendidx])

processed = {}
processed['t'] = t
processed['W'] = W
processed['Lzu'] = Lzu
processed['KE'] = KE
processed['EN'] = EN
processed['ENbdry'] = ENbdry
processed['PA'] = PA
processed['PAbdry1'] = PAbdry1
processed['PAbdry2'] = PAbdry2

processed['W_tavg'] = W_tavg
processed['Lzu_tavg'] = Lzu_tavg
processed['KE_tavg'] = KE_tavg
processed['EN_tavg'] = EN_tavg
processed['KE_growth_pred'] = (eps/(np.pi)) * t # these are predicted for area-average, i.e., 1/pi * K_tot 
processed['KE_tavg_expected'] = KE_tavg_expected

processed['ENbdry_tavg'] = ENbdry_tavg
processed['PA_tavg'] = PA_tavg
processed['PAbdry1_tavg'] = PAbdry1_tavg
processed['PAbdry2_tavg'] = PAbdry2_tavg

print('saving output')
np.save('processed_vm_scalars_' + output_suffix + '.npy', processed)

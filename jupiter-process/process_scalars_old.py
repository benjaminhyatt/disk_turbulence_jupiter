import numpy as np
import h5py

Nphi, Nr = 1536, 768 #512, 256 #1024, 512 #512, 256 #640, 320 #768, 384 #512, 256 #1024, 512 
nu = 4e-5 #2e-4 #4e-5 #2e-4 #8e-5 #1e-3 #2e-4 #1e-4 #2e-4 #8e-5 #5e-5
gamma = 2500 #1920 #400 #240 #2372 #1920 #240 #675 #85 #0 #1920 #240 #30
k_force = 40 #20 #40 #20 #80 #20 #20 #70 #35 #20 #50

alpha = 3.3e-2 #1e-2
amp = 1

ring = 0

restart_evolved = False #False #True

restart_hyst = False #True
hystn = 7

old = False
if old:
    eps = 2 * amp**2
else:
    eps = amp**2

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
#output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) 
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

if restart_hyst:
    output_suffix += '_restart_hyst_{:d}'.format(hystn)


# load in analysis
print("loading primary run")
f1 = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s2.h5')
#f1 = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
#f1 = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
t = f1['tasks/KE'].dims[0]['sim_time'][:]

#Lzu = f1['tasks/Lzu'][:, 0, 0]
KE = f1['tasks/KE'][:, 0, 0]
EN = f1['tasks/EN'][:, 0, 0]
#print(f1['tasks/KE'])
#ENbdry = f1['tasks/ENbdry'][:, 0, 0]
#PA = f1['tasks/PA'][:, 0, 0]
#PAbdry1 = f1['tasks/PAbdry1'][:, 0, 0]
#PAbdry2 = f1['tasks/PAbdry2'][:, 0, 0]


if old: # used Integrate, want avg
    #Lzu *= 1/(np.pi)
    KE *= 1/(np.pi)
    EN *= 1/(np.pi)

# temporal averages
tdur = 5 #50 
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

KE_tavg = np.mean(KE[tstartidx:tendidx])
EN_tavg = np.mean(EN[tstartidx:tendidx])
KE_tavg_expected = ((eps / np.pi) - nu * EN_tavg) / (2 * alpha)
#print(KE_tavg, EN_tavg)
#print(KE_tavg_expected)
processed = {}
processed['t'] = t
#processed['Lzu'] = Lzu
processed['KE'] = KE
processed['EN'] = EN
#processed['ENbdry'] = ENbdry
#processed['PA'] = PA
#processed['PAbdry1'] = PAbdry1
#processed['PAbdry2'] = PAbdry2
processed['KE_tavg'] = KE_tavg
processed['EN_tavg'] = EN_tavg
processed['KE_growth_pred'] = (eps/(np.pi)) * t # these are predicted for area-average, i.e., 1/pi * K_tot 
processed['KE_tavg_expected'] = KE_tavg_expected


print('saving output')
np.save('processed_scalars_' + output_suffix + '.npy', processed)

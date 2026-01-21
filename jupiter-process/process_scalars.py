"""
Usage:
    process_scalars.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the output file [default: processed_scalars]
"""


import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)

def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2]) # if str begins with format XdY
    except:
        sec = 0
    exp = int(a[-2:])
    return (first + sec/10) * 10**(exp)

print("args read in")
print(args)

output_prefix = args['--output']

file_str = args['<file>'][0]
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
print(output_suffix)

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)

alpha_vals = np.array((1e-2, 3.3e-2))
eps_vals = np.array([1.0, 2.0])
nu_vals = np.array([2e-4, 8e-5, 4e-5, 2e-5])

alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]
print(alpha, eps, nu)


# load in analysis
f1 = h5py.File(file_str)
t = f1['tasks/KE'].dims[0]['sim_time'][:]


#Lzu = f1['tasks/Lzu'][:, 0, 0]
KE = f1['tasks/KE'][:, 0, 0]
EN = f1['tasks/EN'][:, 0, 0]
#print(f1['tasks/KE'])
#ENbdry = f1['tasks/ENbdry'][:, 0, 0]
#PA = f1['tasks/PA'][:, 0, 0]
#PAbdry1 = f1['tasks/PAbdry1'][:, 0, 0]
#PAbdry2 = f1['tasks/PAbdry2'][:, 0, 0]

tdur = 30 #100 
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

KE_tavg = np.mean(KE[tstartidx:tendidx])
EN_tavg = np.mean(EN[tstartidx:tendidx])
KE_tavg_expected = ((eps / np.pi) - nu * EN_tavg) / (2 * alpha)
print(EN_tavg)
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
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

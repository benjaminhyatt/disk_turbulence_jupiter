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
    if a[-3] == 'p':
        sgn = 1 
    else:
        sgn = -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

print("args read in")
print(args)

output_prefix = args['--output']

file_str = args['<file>'][0]
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)
alpha_vals = np.array((1e-2, 3.3e-2, 1e-1))
eps_vals = np.array([1.0, 2.0, 3.0])
nu_vals = np.array([2e-4, 5e-5, 8e-4, 8/90000, 4e-4])

alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]

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

W = f1['tasks/W'][:, 0, 0]
om0 = f1['tasks/om0'][:, 0, 0]
om1c = f1['tasks/om1c'][:, 0, 0]
om1s = f1['tasks/om1s'][:, 0, 0] 

nu0 = f1['tasks/nu0'][:, 0, 0]
nu1c = f1['tasks/nu1c'][:, 0, 0]
nu1s = f1['tasks/nu1s'][:, 0, 0]

tdur =  1.5/alpha
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

#tdur = 150
#tend = 400
#tendidx = np.where(t >= tend)[0][0]
#tstartidx = np.where(t >= tend - tdur)[0][0]
#tstart = t[tstartidx]

KE_tavg = np.mean(KE[tstartidx:tendidx])
EN_tavg = np.mean(EN[tstartidx:tendidx])
KE_tavg_expected = ((eps / np.pi) - nu * EN_tavg) / (2 * alpha)

processed = {}
processed['t'] = t
#processed['Lzu'] = Lzu
processed['KE'] = KE
processed['EN'] = EN
#processed['ENbdry'] = ENbdry
#processed['PA'] = PA
#processed['PAbdry1'] = PAbdry1
#processed['PAbdry2'] = PAbdry2

processed['om0'] = om0
processed['W'] = W
processed['om1c'] = om1c
processed['om1s'] = om1s

processed['nu0'] = nu0
processed['nu1c'] = nu1c
processed['nu1s'] = nu1s

processed['KE_tavg'] = KE_tavg
processed['EN_tavg'] = EN_tavg
processed['KE_growth_pred'] = (eps/(np.pi)) * t # these are predicted for area-average, i.e., 1/pi * K_tot 
processed['KE_tavg_expected'] = KE_tavg_expected

print('saving output')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

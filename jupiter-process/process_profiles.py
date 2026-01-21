"""
Usage:
    process_profiles.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the output file [default: processed_profiles]
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

vortm0 = f1['tasks/vortm0'][:, 0, :]
pvortm0 = f1['tasks/pvortm0'][:, 0, :]
drvortm0 = f1['tasks/drvortm0'][:, 0, :]
drpvortm0 = f1['tasks/drpvortm0'][:, 0, :]
dr2pvortm0 = f1['tasks/dr2pvortm0'][:, 0, :]

tdur = 10
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

vortm0_tavg = np.mean(vortm0[tstartidx:tendidx, :], axis = 0)
pvortm0_tavg = np.mean(pvortm0[tstartidx:tendidx, :], axis = 0)
drvortm0_tavg = np.mean(drvortm0[tstartidx:tendidx, :], axis = 0)
drpvortm0_tavg = np.mean(drpvortm0[tstartidx:tendidx, :], axis = 0)
dr2pvortm0_tavg = np.mean(dr2pvortm0[tstartidx:tendidx, :], axis = 0)

processed = {}
processed['t'] = t
processed['vortm0'] = vortm0
processed['vortm0_tavg'] = vortm0_tavg
processed['pvortm0'] = vortm0
processed['pvortm0_tavg'] = vortm0_tavg
processed['drvortm0'] = vortm0
processed['drvortm0_tavg'] = vortm0_tavg
processed['drpvortm0'] = vortm0
processed['drpvortm0_tavg'] = vortm0_tavg
processed['dr2pvortm0'] = vortm0
processed['dr2pvortm0_tavg'] = vortm0_tavg

print("Saving output")
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

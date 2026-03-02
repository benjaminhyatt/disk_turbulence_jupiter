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
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
alpha_vals = np.array((1e-2, 3.3e-2))
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]

# load in profile data
f1 = h5py.File(file_str)
t = np.array(f1['tasks/KE'].dims[0]['sim_time'][:])

ws = np.arange(len(t))
nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0

# radial profiles 
vortm0 = np.array(f1['tasks/vortm0'][:, 0, :])
pvortm0 = np.array(f1['tasks/pvortm0'][:, 0, :])
drvortm0 = np.array(f1['tasks/drvortm0'][:, 0, :])
drpvortm0 = np.array(f1['tasks/drpvortm0'][:, 0, :])
dr2pvortm0 = np.array(f1['tasks/dr2pvortm0'][:, 0, :])

# take time averages
tdur = 1.5/alpha
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

processed['nw'] = nw
processed['ws'] = ws
processed['tw'] = tw

processed['r'] = np.array(f1['tasks/vortm0'].dims[2][0])
processed['vortm0'] = vortm0
processed['vortm0_tavg'] = vortm0_tavg
processed['pvortm0'] = pvortm0
processed['pvortm0_tavg'] = pvortm0_tavg
processed['drvortm0'] = drvortm0
processed['drvortm0_tavg'] = drvortm0_tavg
processed['drpvortm0'] = drpvortm0
processed['drpvortm0_tavg'] = drpvortm0_tavg
processed['dr2pvortm0'] = dr2pvortm0
processed['dr2pvortm0_tavg'] = dr2pvortm0_tavg

print("Saving output")
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

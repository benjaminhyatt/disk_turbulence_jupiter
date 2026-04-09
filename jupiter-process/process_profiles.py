"""
Usage:
    process_profiles.py <file>... [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_profiles]
    --t_start=<float>       if provided, the time to begin the t_avg window [default: None]    
    --t_end=<float>         if provided, the time to end the t_avg window [default: None]
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

t_start = eval(args['--t_start'])
t_end = eval(args['--t_end'])

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

Nr = vortm0.shape[1]
print(Nr)
Nphi = int(2*Nr)

try: 
    um0 = np.array(f1['tasks/um0'][:, 0, :])
except:
    import dedalus.public as d3
    dealias = 3/2
    dtype = np.float64

    coords = d3.PolarCoordinates('phi', 'r')
    dist = d3.Distributor(coords, dtype=dtype)
    disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
    ephi = dist.VectorField(coords, bases=disk) #Vector ephi
    ephi['g'][0] = 1 
    radial_basis = disk.radial_basis
 
    u = dist.VectorField(coords, name='u', bases=disk)
    #um0r = dist.Field(bases=radial_basis)
    #um0 = np.zeros((nw, Nr))
    um0 = np.zeros((nw, int(dealias*Nr)))
    for w in range(nw):
        u.load_from_hdf5(f1, w)
        u.change_scales(dealias)
        uphi = u@ephi
        um0r = d3.Average(uphi, coords['phi']).evaluate()
        #um0r.change_scales(1)
        if w % 10 == 0:
            print(w, nw)
        um0[w, :] = np.copy(um0r['g'])


# take time averages
if (t_start is not None) and (t_end is not None):
    try:
        tendidx = np.where(t >= t_end)[0][0]
        tend = t[tendidx]
        tstartidx = np.where(t >= t_start)[0][0]
        tstart = t[tstartidx]
    except:
        print("Provided t_start and t_end not compatible with sim times, trying default")
        print("t", t)
        tdur = 1.5/alpha
        tendidx = -1
        tend = t[tendidx]
        tstartidx = np.where(t >= tend - tdur)[0][0]
        tstart = t[tstartidx] 
else:
    tdur = 1.5/alpha
    tendidx = -1
    tend = t[tendidx]
    tstartidx = np.where(t >= tend - tdur)[0][0]
    tstart = t[tstartidx]

um0_tavg = np.mean(um0[tstartidx:tendidx, :], axis = 0)
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
processed['um0'] = um0
processed['um0_tavg'] = um0_tavg
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

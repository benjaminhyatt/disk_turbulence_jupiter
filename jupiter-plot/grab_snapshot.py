"""
Usage:
    grab_snapshot.py <file>... [options]

Options:    
    --time=<float>  time of write to grab [default: None]
    --write=<int>   write to grab (if both time and write are not None, time will win-out) [default: None]
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
### read arguments passed to script ###
print("args read in")
print(args)

file_str = args['<file>'][0]
time = args['--time']
write = args['--write']
if time != 'None':
    time = float(time)
if write != 'None':
    write = int(write)

### read analysis file ###
f = h5py.File(file_str)

### string parsing to identify parameters ###
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

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
gamma_read = str_to_float(gamma_str)
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]

# load in analysis file
f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

if time is not None:
    wout = np.where(t >= time)[0][0]
elif write is not None:
    if write <= t.shape[0] - 1:
        wout = write
    else:
        print("Invalid choice of write for t of shape", t.shape)
else:
    print("Grabbing last write by default")
    wout = -1

# dedalus setup 
dealias = 3/2
Nphi_deal = int(np.round(dealias * Nphi))
Nr_deal = int(np.round(dealias * Nr))
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

# fields to grab
u = dist.VectorField(coords, name = 'u', bases = disk)
vort = dist.Field(name = 'vort', bases = disk)

# load data from corresponding write
u.load_from_hdf5(f, wout)
vort.load_from_hdf5(f, wout)

ug = np.copy(u['g'])
uc = np.copy(u['c'])
vortg = np.copy(vort['g'])
vortc = np.copy(vort['c'])

grab = {}
grab['wout'] = wout
grab['t'] = t
grab['phi'] = phi
grab['r'] = r
grab['phi_deal'] = phi_deal
grab['r_deal'] = r_deal
grab['ug'] = ug 
grab['uc'] = uc
grab['vortg'] = vortg 
grab['vortc'] = vortc

print('saving data grab as: ' + 'grab_' + str(wout) + '_' + output_suffix + '.npy')
np.save('grab_' + str(wout) + '_' + output_suffix + '.npy', grab)

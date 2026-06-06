"""
Usage:
    process_profiles_filtered.py <ivp_file> <tracking_file> [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_profiles]
"""

import numpy as np
import dedalus.public as d3
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

#t_start = eval(args['--t_start'])
#t_end = eval(args['--t_end'])

file_str = args['<ivp_file>']
tracking_file_str = args['<tracking_file>']
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
alpha_vals = np.array((1e-2, 3.3e-2))
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])

dealias = 3/2
dtype = np.float64

coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
ephi = dist.VectorField(coords, bases=disk) #Vector ephi
ephi['g'][0] = 1 
rscalar = dist.Field(bases=disk)
rscalar.change_scales(dealias)
rscalar['g'] = r_deal
radial_basis = disk.radial_basis

u = dist.VectorField(coords, name='u', bases=disk)
vort = dist.Field(name='vort', bases=disk)

Omega = dist.Field(name='Omega', bases=disk)    
uphim0 = dist.Field(bases=radial_basis)

phi_mesh, r_mesh = np.meshgrid(phi_deal[:, 0], r_deal[0, :])
phi_mesh = phi_mesh.T
r_mesh = r_mesh.T

# load in profile data
f1 = h5py.File(file_str)
t = np.array(f1['tasks/KE'].dims[0]['sim_time'][:])
ws = np.arange(len(t))
nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0
# radial profiles 
#vortm0_in = np.array(f1['tasks/vortm0'][:, 0, :])
pvortm0_in = np.array(f1['tasks/pvortm0'][:, 0, :])
#drvortm0_in = np.array(f1['tasks/drvortm0'][:, 0, :])
drpvortm0_in = np.array(f1['tasks/drpvortm0'][:, 0, :])
dr2pvortm0_in = np.array(f1['tasks/dr2pvortm0'][:, 0, :])

# load in tracking results
tracking_in = np.load(tracking_file_str, allow_pickle=True)[()]
phi_locs = tracking_in['phi_locs']
r_locs = tracking_in['r_locs']
ws_tracked = tracking_in['ws']
nw_tracked = tracking_in['nw']
print("nw", nw, "tracked", nw_tracked)
print("ws first and last", ws[0], ws[-1], "tracked", ws_tracked[0], ws_tracked[-1])
nw = nw_tracked
ws = ws_tracked

t_start = tracking_in['tw'][0]
t_end = tracking_in['tw'][-1]

um0 = np.zeros((nw, int(dealias*Nr)))
#pvortm0 = np.zeros((nw, int(dealias*Nr)))
#drpvortm0 = np.zeros((nw, int(dealias*Nr)))
#dr2pvortm0 = np.zeros((nw, int(dealias*Nr)))

#for w in range(nw):
print(ws_tracked)
for k, w in enumerate(ws_tracked):
    # load field(s)
    u.load_from_hdf5(f1, w) 
    #vort.load_from_hdf5(f1, w)

    # find vortex in vorticity
    lon_loc = phi_locs[k]
    r_loc = r_locs[k] # not currently using

    # apply azimuthal filter
    zero_mask_1 = np.logical_and(phi_mesh > np.max((0, lon_loc - np.pi/2)), phi_mesh < np.min((2*np.pi, lon_loc + np.pi/2)))
    zero_mask_2 = np.logical_and(phi_mesh > (lon_loc - np.pi/2) % 2 * np.pi , phi_mesh < (lon_loc + np.pi/2) % 2 * np.pi)
    zero_mask = np.logical_or(zero_mask_1, zero_mask_2)    
    
    uphi = (u @ ephi).evaluate()
    uphi.change_scales(dealias)
    uphi['g'][zero_mask] *= 0.
    uphim0.change_scales(dealias)
    uphim0['g'] = 2 * d3.Average(uphi, 'phi').evaluate()['g']
    um0[k, :] = np.copy(uphim0['g'])

    # possibly better for regularity?
    #Omega = (u@ephi)/rscalar
    #Omegam0r = d3.Average(Omega, coords['phi']).evaluate()
    #uphi.change_scales(dealias)
    #Omegam0r.change_scales(dealias)
    # back to velocity
    #uphi['g'] = Omegam0r['g'] * r_deal
    #um0[w, :] = np.copy(uphi['g'])

    if k % 10 == 0:
        print(k, nw)

# take time averages
#if (t_start is not None) and (t_end is not None):
#    try:
#tendidx = np.where(t >= t_end)[0][0]
#tend = t[tendidx]
#tstartidx = np.where(t >= t_start)[0][0]
#tstart = t[tstartidx]
#    except:
#        print("Provided t_start and t_end not compatible with sim times, trying default")
#        print("t", t)
#        tdur = 1.5/alpha
#        tendidx = -1
#        tend = t[tendidx]
#        tstartidx = np.where(t >= tend - tdur)[0][0]
#        tstart = t[tstartidx] 
#else:
#    tdur = 1.5/alpha
#    tendidx = -1
#    tend = t[tendidx]
#    tstartidx = np.where(t >= tend - tdur)[0][0]
#    tstart = t[tstartidx]

#print(t_start, t_end, tstart, tend, tstartidx, tendidx)

#um0_tavg = np.mean(um0[tstartidx:tendidx, :], axis = 0)
#vortm0_tavg = np.mean(vortm0[tstartidx:tendidx, :], axis = 0)
#pvortm0_tavg = np.mean(pvortm0[tstartidx:tendidx, :], axis = 0)
#drvortm0_tavg = np.mean(drvortm0[tstartidx:tendidx, :], axis = 0)
#drpvortm0_tavg = np.mean(drpvortm0[tstartidx:tendidx, :], axis = 0)
#dr2pvortm0_tavg = np.mean(dr2pvortm0[tstartidx:tendidx, :], axis = 0)

um0_tavg = np.mean(um0, axis=0)

processed = {}

processed['nw'] = nw
processed['ws'] = ws
processed['tw'] = tw

processed['r'] = np.array(f1['tasks/vortm0'].dims[2][0])
processed['r_deal'] = r_deal
processed['um0'] = um0
processed['um0_tavg'] = um0_tavg
#processed['Omegam0'] = Omegam0
#processed['Omegam0_tavg'] = Omegam0_tavg
#processed['vortm0'] = vortm0
#processed['vortm0_tavg'] = vortm0_tavg
#processed['pvortm0'] = pvortm0
#processed['pvortm0_tavg'] = pvortm0_tavg
#processed['drvortm0'] = drvortm0
#processed['drvortm0_tavg'] = drvortm0_tavg
#processed['drpvortm0'] = drpvortm0
#processed['drpvortm0_tavg'] = drpvortm0_tavg
#processed['dr2pvortm0'] = dr2pvortm0
#processed['dr2pvortm0_tavg'] = dr2pvortm0_tavg

print("Saving output")
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

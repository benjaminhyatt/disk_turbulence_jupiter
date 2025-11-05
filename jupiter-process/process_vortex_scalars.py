"""
Track positions of maxima and minima in the vorticity and potential vorticity. 

Usage:
    process_vortex_scalars.py [--r_cut=<float>]

Options:
    --r_cut=<float>  exclude field data from outside of this radius, e.g., to ignore zonal jets [default: 1.0] 
"""

import numpy as np
import h5py 
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import glob

from docopt import docopt
args = docopt(__doc__)

if args['--r_cut'] is not None:
    r_cutoff = float(args['--r_cut'])
else:
    r_cutoff = 1.0

Nphi, Nr = 512, 256 #768, 384 #512, 256 #1024, 512 
nu = 2e-4 #8e-5 #5e-5 #2e-4
gamma = 85 #85 #0 #1920 #240 #30
k_force = 20 #20 #50
alpha = 1e-2
amp = 1

old = False #False #True
restart_evolved = False #True

if old:
    eps = 2 * amp**2
else:
    eps = amp**2
u_ea  = np.sqrt(eps/alpha)
if gamma != 0.:
    Lgam_u_ea = (u_ea / gamma) ** (1/3)
else:
    Lgam_u_ea = np.nan

dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
phi, r = dist.local_grids(disk, scales=(1, 1))
G = - 0.5 * gamma * pow(r, 2)

if r_cutoff > r[0, -1]:
    r_cutoff_idx = -1
else:
    r_cutoff_idx = np.where(r[0, :] >= r_cutoff)[0][0]

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

profiles_dir = 'analysis_' + output_suffix
profiles_file = glob.glob('../jupiter-run/' + profiles_dir + '/*.h5')

logger.info("loading primary run")
f = h5py.File(profiles_file[0], mode='r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]

# main dsets
dset_vort = f['tasks/vorticity']
vort = np.array(dset_vort)
#vort = np.copy(np.array(dset_vort[:, :, :]))
if gamma != 0.:
    dset_pvort = f['tasks/pv']
    pvort = np.array(dset_pvort) #np.copy(np.array(dset_pvort[:, :, :]))
else:
    pvort = vort #np.copy(vort)
# other
dset_KE = f['tasks/KE'] 
KE = np.array(dset_KE[:, 0, 0])#np.copy(np.array(dset_KE[:, 0, 0])) 
dset_u = f['tasks/u']
u = np.array(dset_u)
#u = np.copy(np.array(dset_u[:, :, :, :]))

tasks_out = ['Lgamma', 'vort_extrema', 'pvort_extrema']

subtasks_out = {}
subtasks_out['Lgamma'] = ['Lgamma_u_rms', 'Lgamma_u_rms_tavg', 'Lgamma_u_max', 'Lgamma_u_max_tavg', 'Lgamma_u_ea_tavg']
subtasks_out['vort_extrema'] = ['vort_max', 'vort_max_tavg',  'vort_min', 'vort_min_tavg']
subtasks_out['pvort_extrema'] = ['pvort_max', 'pvort_max_tavg', 'pvort_min', 'pvort_min_tavg']

labels_out = {}
labels_out['Lgamma'] = [r'$r = \left(|u|_{\rm rms} / \gamma \right)^{1/3}$', r'$\langle r = \left(|u|_{\rm rms} / \gamma \right)^{1/3}\rangle$', r'$r = \left(|u|_{\rm max} / \gamma \right)^{1/3}$', r'$\langle r = \left(|u|_{\rm max} / \gamma \right)^{1/3}\rangle$', r'$r = \left(\sqrt{\epsilon/\alpha} / \gamma \right)^{1/3}$']
labels_out['vort_extrema'] = [r'$r_{\omega_{\rm max}}$', r'$\langle r_{\omega_{\rm max}}\rangle$', r'$r_{\omega_{\rm min}}$', r'$\langle r_{\omega_{\rm min}}\rangle$']
labels_out['pvort_extrema'] = [r'$r_{q_{\rm max}}$', r'$\langle r_{q_{\rm max}}\rangle$', r'$r_{q_{\rm min}}$', r'$\langle r_{q_{\rm min}}\rangle$']

# processing
u_rmss = np.zeros(nwritesf)
u_maxs = np.zeros(nwritesf)

Lgams_u_rms = np.zeros(nwritesf)
Lgams_u_max = np.zeros(nwritesf)

phi_vort_maxs = np.zeros(nwritesf)
phi_vort_mins = np.zeros(nwritesf)
phi_pvort_maxs = np.zeros(nwritesf)
phi_pvort_mins = np.zeros(nwritesf)
r_vort_maxs = np.zeros(nwritesf)
r_vort_mins = np.zeros(nwritesf)
r_pvort_maxs = np.zeros(nwritesf)
r_pvort_mins = np.zeros(nwritesf)


progress_cad = np.ceil(nwritesf / 50)
logger.info("Entering processing loop")
for w in range(nwritesf):
    if old: 
        u_rms = np.sqrt(2 * KE[w] / np.pi) # KE from Integrate
    else:
        u_rms = np.sqrt(2 * KE[w]) # KE from Average
   
    u_mag = np.sqrt(u[w, 0, :, :]**2 + u[w, 1, :, :]**2)
    u_max = np.max(u_mag)

    if gamma != 0.:
        Lgam_u_rms = (u_rms / gamma) ** (1/3) 
        Lgam_u_max = (u_max / gamma) ** (1/3)
    else: 
        Lgam_u_rms = np.nan
        Lgam_u_max = np.nan

    vort_max, vort_min = (np.max(vort[w, :, :r_cutoff_idx]), np.min(vort[w, :, :r_cutoff_idx]))
    pvort_max, pvort_min = (np.max(pvort[w, :, :r_cutoff_idx]), np.min(pvort[w, :, :r_cutoff_idx]))

    if w > 0:
        phiidx_vort_max, phiidx_vort_min = (np.where(vort[w] == vort_max)[0][0], np.where(vort[w] == vort_min)[0][0])
        ridx_vort_max, ridx_vort_min = (np.where(vort[w] == vort_max)[1][0], np.where(vort[w] == vort_min)[1][0])
        phi_vort_max, phi_vort_min = (phi[phiidx_vort_max, 0], phi[phiidx_vort_min, 0])
        r_vort_max, r_vort_min = (r[0, ridx_vort_max], r[0, ridx_vort_min])
        phiidx_pvort_max, phiidx_pvort_min = (np.where(pvort[w] == pvort_max)[0][0], np.where(pvort[w] == pvort_min)[0][0])
        ridx_pvort_max, ridx_pvort_min = (np.where(pvort[w] == pvort_max)[1][0], np.where(pvort[w] == pvort_min)[1][0])
        phi_pvort_max, phi_pvort_min = (phi[phiidx_pvort_max, 0], phi[phiidx_pvort_min, 0])
        r_pvort_max, r_pvort_min = (r[0, ridx_pvort_max], r[0, ridx_pvort_min])
    else:
        phi_vort_max, phi_vort_min = (0., 0.)
        r_vort_max, r_vort_min = (0., 0.)
        phi_pvort_max, phi_pvort_min = (0., 0.)
        r_pvort_max, r_pvort_min = (0., 0.)

    u_rmss[w] = u_rms
    u_maxs[w] = u_max
    Lgams_u_rms[w] = Lgam_u_rms
    Lgams_u_max[w] = Lgam_u_max
    
    phi_vort_maxs[w] = phi_vort_max
    phi_vort_mins[w] = phi_vort_min
    phi_pvort_maxs[w] = phi_pvort_max
    phi_pvort_mins[w] = phi_pvort_min
    r_vort_maxs[w] = r_vort_max
    r_vort_mins[w] = r_vort_min
    r_pvort_maxs[w] = r_pvort_max
    r_pvort_mins[w] = r_pvort_min

    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

# calculate temporal averages
t = dset_vort.dims[0]['sim_time'][:]
tdur = 50 #t[-1] * 0.2
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

Lgam_u_rms_tavg = np.nanmean(Lgams_u_rms[tstartidx:tendidx])
Lgam_u_max_tavg = np.nanmean(Lgams_u_max[tstartidx:tendidx])

r_vort_max_tavg = np.nanmean(r_vort_maxs[tstartidx:tendidx])
r_vort_min_tavg = np.nanmean(r_vort_mins[tstartidx:tendidx])
r_pvort_max_tavg = np.nanmean(r_pvort_maxs[tstartidx:tendidx])
r_pvort_min_tavg = np.nanmean(r_pvort_mins[tstartidx:tendidx])

# store 
processed = {}
for taskout in tasks_out:
    processed[taskout] = {}
    processed[taskout]['t'] = np.array(dset_vort.dims[0]['sim_time'])
    if taskout == 'Lgamma':
        processed[taskout]['data_Lgamma_u_rms'] = Lgams_u_rms
        processed[taskout]['data_Lgamma_u_rms_tavg'] = np.ones(nwritesf) * Lgam_u_rms_tavg
        processed[taskout]['data_Lgamma_u_max'] = Lgams_u_max
        processed[taskout]['data_Lgamma_u_max_tavg'] = np.ones(nwritesf) * Lgam_u_max_tavg
        processed[taskout]['data_Lgamma_u_ea_tavg'] = Lgam_u_ea
    
    if taskout == 'vort_extrema':
        # only for other processing
        processed[taskout]['phi_vort_max'] = phi_vort_maxs
        processed[taskout]['phi_vort_min'] = phi_vort_mins
        # for plotting and other processing    
        processed[taskout]['data_vort_max'] = r_vort_maxs
        processed[taskout]['data_vort_min'] = r_vort_mins
        processed[taskout]['data_vort_max_tavg'] = np.ones(nwritesf) * r_vort_max_tavg
        processed[taskout]['data_vort_min_tavg'] = np.ones(nwritesf) * r_vort_min_tavg
    
    if taskout == 'pvort_extrema':
        # only for other processing
        processed[taskout]['phi_pvort_max'] = phi_pvort_maxs
        processed[taskout]['phi_pvort_min'] = phi_pvort_mins
        # for plotting and other processing
        processed[taskout]['data_pvort_max'] = r_pvort_maxs
        processed[taskout]['data_pvort_min'] = r_pvort_mins
        processed[taskout]['data_pvort_max_tavg'] = np.ones(nwritesf) * r_pvort_max_tavg
        processed[taskout]['data_pvort_min_tavg'] = np.ones(nwritesf) * r_pvort_min_tavg

f.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

processed['tendidx'] = tendidx
processed['tend'] = tend
processed['tstartidx'] = tstartidx
processed['tstart'] = tstart

print("Saving output")
np.save('processed_vortex_scalars_' + output_suffix + '.npy', processed)

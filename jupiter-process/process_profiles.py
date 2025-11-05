#### need to run process_scalars.py first, if you want to load in u_rms calculated there...

import numpy as np
import h5py 
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import glob

Nphi, Nr = 1024, 512 #640, 320 #768, 384 #512, 256 #1024, 512 
nu = 8e-5 #1e-3 #2e-4 #1e-4 #2e-4 #8e-5 #5e-5
gamma = 0 #675 #85 #0 #1920 #240 #30
k_force = 20 #20 #70 #35 #20 #50

alpha = 1e-2
amp = 1 

ring = 0 

restart_evolved = False #False #True
old = False #True #False

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

dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
phi, r = dist.local_grids(disk, scales=(1, 1))
G = - 0.5 * gamma * pow(r, 2)

processed_scalars = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle = True)[()]
r_Lg = (processed_scalars['u_rms_tavg'] / gamma)**(1/3)

profiles_dir = 'analysis_' + output_suffix
profiles_file = glob.glob('../jupiter-run/' + profiles_dir + '/*.h5')

logger.info("loading primary run")
f = h5py.File(profiles_file[0], mode='r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]
 
dset_vort_m0 = f['tasks/w0']
vort_m0 = np.copy(np.array(dset_vort_m0[:, 0, :]))
uphi_m0 = np.copy(np.array(f['tasks/um0'][:, 0, 0, :]))
pvort_m0 = vort_m0 + G

tasks_in = ['um0', 'w0']
tasks_out = ['profiles_u', 'profiles_vort']

subtasks_out = {}
subtasks_out['profiles_u'] = ['uphi_m0', 'uphi_m0_tavg'] 
subtasks_out['profiles_vort'] = ['vort_m0', 'vort_m0_tavg', 'pvort_m0', 'pvort_m0_tavg']

labels_out = {}
labels_out['profiles_u'] = [r'$u_{\phi, m = 0}(r)$', r'$\langle u_{\phi, m=0}(r) \rangle$']
labels_out['profiles_vort'] = [r'$\omega_{m=0}(r)$', r'$\langle \omega_{m=0}(r) \rangle$', r'$q_{m=0}$', r'$\langle q_{m=0}(r)\rangle$']

processed = {}

# temporal averages
t = dset_vort_m0.dims[0]['sim_time'][:]
tdur = 5
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

uphi_m0_tavg = np.mean(uphi_m0[tstartidx:, :], axis = 0)
vort_m0_tavg = np.mean(vort_m0[tstartidx:, :], axis = 0)
pvort_m0_tavg = np.mean(pvort_m0[tstartidx:, :], axis = 0)

ymin_u = np.min(uphi_m0_tavg)
ymin_vort = np.min(vort_m0_tavg)
ymin_vort = np.min(np.concatenate(([ymin_vort], pvort_m0_tavg)))
ymax_u = np.max(uphi_m0_tavg)
ymax_vort = np.max(vort_m0_tavg)
ymax_vort = np.max(np.concatenate(([ymax_vort], pvort_m0_tavg)))

idx = 0
progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
        processed[idx + w][taskout]['t'] = np.array(dset_vort_m0.dims[0]['sim_time'])[w]
        processed[idx + w][taskout]['r'] = np.array(dset_vort_m0.dims[2][0])
        
        if taskout == 'profiles_u':
            processed[idx + w][taskout]['data_uphi_m0'] = uphi_m0[w]
            processed[idx + w][taskout]['data_uphi_m0_tavg'] = uphi_m0_tavg
        if taskout == 'profiles_vort':
            processed[idx + w][taskout]['data_vort_m0'] = vort_m0[w]
            processed[idx + w][taskout]['data_pvort_m0'] = pvort_m0[w]
            processed[idx + w][taskout]['data_vort_m0_tavg'] = vort_m0_tavg
            processed[idx + w][taskout]['data_pvort_m0_tavg'] = pvort_m0_tavg

    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

f.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

processed['r_Lg'] = r_Lg

processed['tendidx'] = tendidx
processed['tend'] = tend
processed['tstartidx'] = tstartidx
processed['tstart'] = tstart

processed['ymin_u'] = ymin_u
processed['ymax_u'] = ymax_u
processed['ymin_vort'] = ymin_vort
processed['ymax_vort'] = ymax_vort

print("Saving output")
np.save('processed_profiles_' + output_suffix + '.npy', processed)

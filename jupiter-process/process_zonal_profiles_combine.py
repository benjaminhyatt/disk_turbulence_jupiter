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
er = dist.VectorField(coords, bases=disk) #Vector er
er['g'][1] = 1

disk_field = dist.Field(name = 'disk_f', bases = disk)

profiles_dir = 'analysis_' + output_suffix
profiles_file = glob.glob('../jupiter-run/' + profiles_dir + '/*.h5')

logger.info("loading primary run")
f = h5py.File(profiles_file[0], mode='r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]

dset_u = f['tasks/u']
dset_vort = f['tasks/vort']

# take derivative(s), evaluate at edge
uphi_edge = np.zeros((nwritesf, Nphi))
dr_uphi_edge = np.zeros((nwritesf, Nphi))
vort_edge = np.zeros((nwritesf, Nphi))

logger.info("Entering processing loop")
progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    
    disk_field.change_scales(1)
    disk_field['g'] = dset_u[w, 0, :, :] # u_phi
    uphi_edge_field = disk_field(r = 1).evaluate()
    uphi_edge_field.change_scales(1)
    uphi_edge[w] = uphi_edge_field['g'][:, 0]

    disk_field.change_scales(1)
    disk_field['g'] = dset_u[w, 0, :, :] # u_phi
    dr_uphi_edge_field = (er@d3.grad(disk_field))(r = 1).evaluate()
    dr_uphi_edge_field.change_scales(1)
    dr_uphi_edge[w] = dr_uphi_edge_field['g'][:, 0] 
    
    disk_field.change_scales(1) 
    disk_field['g'] = dset_vort[w, :, :] # vort
    vort_edge_field = disk_field(r = 1).evaluate()
    vort_edge_field.change_scales(1)
    vort_edge[w] = vort_edge_field['g'][:, 0]

    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

tasks_in = ['u', 'vort']
tasks_out = ['profiles_u', 'profiles_vort']

subtasks_out = {}
subtasks_out['profiles_u'] = ['uphi_edge', 'uphi_edge_tavg', 'dr_uphi_edge', 'dr_uphi_edge_tavg'] 
subtasks_out['profiles_vort'] = ['vort_edge', 'vort_edge_tavg']

labels_out = {}
labels_out['profiles_u'] = [r'$u_{\phi}(r = 1)$', r'$\langle u_{\phi}(r = 1) \rangle$', r'$\partial_r u_{\phi}(r = 1)$', r'$\langle \partial_r u_{\phi}(r = 1) \rangle$']
labels_out['profiles_vort'] = [r'$\omega (r = 1)$', r'$\langle \omega (r = 1) \rangle$']

processed = {}

# temporal averages
t = dset_vort.dims[0]['sim_time'][:]
tdur = 5
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

uphi_edge_tavg = np.mean(uphi_edge[tstartidx:, :], axis = 0)
dr_uphi_edge_tavg = np.mean(dr_uphi_edge[tstartidx:, :], axis = 0)
vort_edge_tavg = np.mean(vort_edge[tstartidx:, :], axis = 0)

ymin_u = np.min(uphi_edge_tavg)
ymin_u2 = np.min(dr_uphi_edge_tavg)
ymin_u = np.min([ymin_u, ymin_u2])
ymin_vort = np.min(vort_edge_tavg)
ymax_u = np.max(uphi_edge_tavg)
ymax_u2 = np.max(dr_uphi_edge_tavg)
ymax_u = np.max([ymax_u, ymax_u2])
ymax_vort = np.max(vort_edge_tavg)

idx = 0
progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
        processed[idx + w][taskout]['t'] = np.array(dset_vort.dims[0]['sim_time'])[w]
        processed[idx + w][taskout]['phi'] = np.array(dset_vort.dims[1][0])
        
        if taskout == 'profiles_u':
            processed[idx + w][taskout]['data_uphi_edge'] = uphi_edge[w]
            processed[idx + w][taskout]['data_uphi_edge_tavg'] = uphi_edge_tavg
            processed[idx + w][taskout]['data_dr_uphi_edge'] = dr_uphi_edge[w]
            processed[idx + w][taskout]['data_dr_uphi_edge_tavg'] = dr_uphi_edge_tavg
        if taskout == 'profiles_vort':
            processed[idx + w][taskout]['data_vort_edge'] = vort_edge[w]
            processed[idx + w][taskout]['data_vort_edge_tavg'] = vort_edge_tavg

    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

f.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

processed['tendidx'] = tendidx
processed['tend'] = tend
processed['tstartidx'] = tstartidx
processed['tstart'] = tstart

processed['ymin_u'] = ymin_u
processed['ymax_u'] = ymax_u
processed['ymin_vort'] = ymin_vort
processed['ymax_vort'] = ymax_vort

print("Saving output")
np.save('processed_zonal_profiles_p2' + output_suffix + '.npy', processed)

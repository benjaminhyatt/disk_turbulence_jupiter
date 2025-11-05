import numpy as np
import h5py 
import dedalus.public as d3
from mpi4py import MPI 
import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nphi, Nr = 1024, 512 #640, 320 #768, 384 #512, 256 #1024, 512 
nu = 2e-4 #5e-4 #2e-4 #1e-4 #2e-4 #8e-5 #5e-5
gamma = 0 #85 #0 #1920 #240 #30
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

f = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
t = np.array(f['tasks/u'].dims[0]['sim_time'])
print(t)
nwritesf = t.shape[0]
# we can choose multiple writes to ensemble average over later
#nwrites = 3
#ws = [1, 250, 1000, 2000]#[1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
#ws = [1, 3000, 3250, 3500, 3750, 4000, 4250, 4500]
#ws = [1, 250, 500] # [1, 250, 1000]
nwrites = 2
ws = [1, 10]
ts = t[ws]

# disk
dealias = 3/2
dtype = np.complex128
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, comm = comm, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_g, r_g = disk.global_grids(dist, scales = (1, 1))
ms = edge.wavenumbers[::2].astype(int) # 0, 1, ..., (Nphi/2 - 1)
ms_pos_idxs = np.arange(1, int(Nphi/2))
mc_pos_idxs = np.arange(2, Nphi, 2) # indices of coeff space corresp to 1, ..., (Nphi/2 - 1)
mc_neg_idxs = np.arange(Nphi-1, 1, -2) #indices of coeff space corresp to -1, ..., -(Nphi/2 - 1)
#mc_pos_idxs = np.arange(1, int(Nphi/2))
#mc_neg_idxs = np.arange(Nphi-1, int(Nphi/2), -1) # arranged in decreasing order -1 to -(Nphi/2 - 1), and skips -(Nphi/2) mode
#mc = np.concatenate(([0], mc_pos_idxs, [int(Nphi/2)], mc_neg_idxs))
Nm = ms.shape[0]

# fields
u = dist.VectorField(coords, name = 'u', bases = disk)
er = dist.VectorField(coords, bases=disk) #Vector er
er['g'][1] = 1 
ephi = dist.VectorField(coords, bases=disk) #Vector ephi
ephi['g'][0] = 1 

# outputs
u_phi_mr = np.zeros((nwrites, Nphi, Nr), dtype = dtype) # uphi, Fourier transform in phi only
u_r_mr = np.zeros((nwrites, Nphi, Nr), dtype = dtype) # ur, Fourier transform in phi only
KE_phi_mr = np.zeros((nwrites, Nm, Nr), dtype = np.float64)
KE_r_mr = np.zeros((nwrites, Nm, Nr), dtype = np.float64)
KE_mr = np.zeros((nwrites, Nm, Nr), dtype = np.float64)
vort_mr = np.zeros((nwrites, Nphi, Nr), dtype = dtype)
EN_mr = np.zeros((nwrites, Nm, Nr), dtype = np.float64)
#KEr_spectrum_mn = np.zeros((nwrites, Nm, Nr)) # ur, Fourier in phi, Bessel in r
#KEphi_spectrum_mn = np.zeros((nwrites, Nm, Nr) # uphi, Fourier in phi, Bessel (Dini) in r

logger.info("entering processing loop")
progress_cad = 1 
for w in range(nwrites):
    if w % progress_cad == 0:
        logger.info("starting to process write (%d / %d)" %(w+1, nwrites))

    #u_in.load_from_hdf5(f, w) # does w = 0 work?
    u.load_from_hdf5(f, ws[w])
    logger.info("write data loaded into field")
    u_phi = ephi @ u
    u_r = er @ u

    # Fourier transform in phi (at each fixed r on grid) -- will later also include radial transforms in this section 
    # iterate over grid values of r
    for ridx, rval in enumerate(r_g[0, :]):
        logger.info("working on radius (%d / %d)" %(ridx , r_g[0, :].shape[0]))

        # velocity
        u_phi_rval = u_phi(r = rval).evaluate()
        u_r_rval = u_r(r = rval).evaluate()
        u_phi_rval.change_scales(1)
        u_r_rval.change_scales(1)
        # gather to rank 0
        u_phi_rval_c = comm.gather(u_phi_rval['c'][:, 0], root = 0)
        u_r_rval_c = comm.gather(u_r_rval['c'][:, 0], root = 0)
        if rank == 0:
            u_phi_mr[w, :, ridx] = np.array(u_phi_rval_c).ravel()
            u_r_mr[w, :, ridx] = np.array(u_r_rval_c).ravel()

            # kinetic energy (total)
            # m = 0
            KE_phi_mr[w, 0, ridx] = 2 * np.pi * (0.5 * u_phi_mr[w, 0, ridx] * np.conj(u_phi_mr[w, 0, ridx])).real
            KE_r_mr[w, 0, ridx] = 2 * np.pi * (0.5 * u_r_mr[w, 0, ridx] * np.conj(u_r_mr[w, 0, ridx])).real
            # m != 0
            KE_phi_mr[w, ms_pos_idxs, ridx] = 2 * np.pi * (0.5 * u_phi_mr[w, mc_pos_idxs, ridx] * np.conj(u_phi_mr[w, mc_pos_idxs, ridx])).real
            KE_phi_mr[w, ms_pos_idxs, ridx] += 2 * np.pi * (0.5 * u_phi_mr[w, mc_neg_idxs, ridx] * np.conj(u_phi_mr[w, mc_neg_idxs, ridx])).real 
            KE_r_mr[w, ms_pos_idxs, ridx] = 2 * np.pi * (0.5 * u_r_mr[w, mc_pos_idxs, ridx] * np.conj(u_r_mr[w, mc_pos_idxs, ridx])).real
            KE_r_mr[w, ms_pos_idxs, ridx] += 2 * np.pi * (0.5 * u_r_mr[w, mc_neg_idxs, ridx] * np.conj(u_r_mr[w, mc_neg_idxs, ridx])).real
            KE_mr[w, :, ridx] = KE_phi_mr[w, :, ridx] + KE_r_mr[w, :, ridx]

        # vorticity
        vort_rval = (-d3.div(d3.skew(u))(r = rval)).evaluate()
        vort_rval.change_scales(1)
        # gather to rank 0
        vort_rval_c = comm.gather(vort_rval['c'][:, 0], root = 0)
        if rank == 0:
            vort_mr[w, :, ridx] = np.array(vort_rval_c).ravel()
            
            # enstrophy (total)
            # m = 0
            EN_mr[w, 0, ridx] = 2 * np.pi * (vort_mr[w, 0, ridx] * np.conj(vort_mr[w, 0, ridx])).real
            # m != 0
            EN_mr[w, ms_pos_idxs, ridx] = 2 * np.pi * (vort_mr[w, mc_pos_idxs, ridx] * np.conj(vort_mr[w, mc_pos_idxs, ridx])).real
            EN_mr[w, ms_pos_idxs, ridx] += 2 * np.pi * (vort_mr[w, mc_neg_idxs, ridx] * np.conj(vort_mr[w, mc_neg_idxs, ridx])).real


# some time averaging
KE_mr_tavg = np.mean(KE_mr[1:, :, :], axis = 0)
EN_mr_tavg = np.mean(EN_mr[1:, :, :], axis = 0)

# save
if rank == 0:
    processed = {}
    processed['nout'] = nwrites
    processed['tout'] = ts
    processed['ws'] = ws
    processed['ms'] = ms
    processed['mc_pos_idxs'] = mc_pos_idxs
    processed['mc_neg_idxs'] = mc_neg_idxs
    processed['rs'] = r_g
    processed['u_phi_mr'] = u_phi_mr
    processed['u_r_mr'] = u_r_mr
    processed['KE_phi_mr'] = KE_phi_mr
    processed['KE_r_mr'] = KE_r_mr
    processed['KE_mr'] = KE_mr
    processed['vort_mr'] = vort_mr
    processed['EN_mr'] = EN_mr

    processed['KE_mr_tavg'] = KE_mr_tavg
    processed['EN_mr_tavg'] = EN_mr_tavg

    np.save('processed_spectra_' + output_suffix + '.npy', processed)



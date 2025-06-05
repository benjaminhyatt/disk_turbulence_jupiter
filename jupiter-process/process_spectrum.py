import numpy as np
import h5py 
import dedalus.public as d3
from mpi4py import MPI 
import logging
logger = logging.getLogger(__name__)

# parallelize over the disk only 
commD = MPI.COMM_WORLD
size = commD.Get_size()
rank = commD.Get_rank()

# no parallelization over Cartesian
commC = MPI.COMM_SELF

# run params
Nphi, Nr = 1024, 512
nu = 5e-5 #2e-4
gamma = 30
k_force = 50

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.0e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
output_suffix = output_suffix.replace('-','m').replace('+','p')
logger.info("loading primary run")
f = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
t = f['tasks/u'].dims[0]['sim_time']
#u = f['tasks/u']
nwritesf = t.shape[0]

# unit disk field
dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, comm = commD, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
#phi, r = dist.local_grids(disk)
phi, r = disk.global_grids(dist, scales = (1, 1))
uD = dist.VectorField(coords, name = 'u', bases = disk)

# unit square cartesian field
Nx, Ny = 256, 256
dtype = np.complex128
coordsC = d3.CartesianCoordinates('x', 'y')
distC = d3.Distributor(coordsC, comm = commC, dtype = dtype)
xbasis = d3.ComplexFourier(coordsC['x'], size = Nx, bounds = (-1, 1), dealias = dealias)
ybasis = d3.ComplexFourier(coordsC['y'], size = Ny, bounds = (-1, 1), dealias = dealias)
#xL, yL = distC.local_grids(xbasis, ybasis)
#x = xbasis.global_grids(distC, scales = [1])[0]
#y = ybasis.global_grids(distC, scales = [1])[0]
x, y = distC.local_grids(xbasis, ybasis) # should be global if comm is COMM_SELF
uC = distC.VectorField(coordsC, name = 'uC', bases = (xbasis, ybasis))

# grid point correspondence
rC = np.zeros((Nx, Ny))
phiC = np.zeros((Nx, Ny))
for i, xi in enumerate(x[:, 0]):
    for j, yj in enumerate(y[0, :]):
        r_ij = np.sqrt(xi**2 + yj**2)
        ang_ij = np.angle(z = xi + 1j * yj)
        phi_ij = ang_ij if (ang_ij > 0) else (ang_ij + 2*np.pi)
        rC[i, j] = r_ij
        phiC[i, j] = phi_ij

# wavenumber setup
kx = xbasis.wavenumbers
ky = ybasis.wavenumbers
nk = kx.shape[0]
kmax = np.max(ky)
klow = np.linspace(0, kmax, nk, endpoint = False)
khigh = klow + klow[1]
kh = np.sqrt(kx[:, None]**2 + ky[None, :]**2) / np.pi

def E(uC_c_ij, kh_ij):
    print(uC_c_ij, kh_ij)
    E_ij = np.real(uC_c_ij[0] * np.conj(uC_c_ij[0]) + uC_c_ij[1] * np.conj(uC_c_ij[1])) # np.real to recast dtype
    E_ij *= 2 * np.pi * (kh_ij**2)
    print(E_ij)
    return E_ij

# wavenumber mask setup
nums = np.zeros(klow.shape[0])
masks = {}
for m in range(klow.shape[0]):
    mask = (kh >= klow[m]) & (kh < khigh[m])
    masks[m] = mask
    nums[m] = np.sum(mask)

# output setup
num_mask = nums > 0
khs = []
for m in range(klow.shape[0]):
    if num_mask[m]:
        khs.append(klow[m]) # representative
khs = np.array(khs)
spectra = np.zeros((nwritesf, khs.shape[0]))

nwritesf = 11 # just to time how long this will take

logger.info("entering processing loop")
progress_cad = 1 #np.ceil(nwritesf / 250)
for w in range(1, nwritesf):
    if w % progress_cad == 0:
        logger.info("(%d / %d) writes processed" %(w, nwritesf))

    # interpolation step (uD -> uC) -- need to do over whole grid
    #uD['g'] = np.copy(np.array(u[w]))
    uD.load_from_hdf5(f, w)
    uD['c'] # this might be redundant, i.e., evaluator might take care of it?

    logger.info("beginning interpolation step")
    for i, xi in enumerate(x[:, 0]):
        for j, yj in enumerate(y[0, :]):
            # only do interpolation if inside disk
            if rC[i, j] < 1:
                print(rC[i, j])
                # interpolation in parallel 
                interp0 = np.copy(uD(r = rC[i, j], phi = phiC[i, j]).evaluate()['g'])
                if rank == 0:
                    interp = interp0[:, 0, 0]
                else:
                    interp = np.empty(2, dtype=np.float64)
                # broadcast result to all ranks
                commD.Bcast(interp, root=0)
                # set value
                near_edge = 0.5 * (1 - np.tanh((rC[i, j] - 0.9)/0.04))
                uC['g'][:, i, j] = near_edge * interp
            else:
                print("else", rC[i, j])
                uC['g'] = 0.


    #for j, yj in enumerate(y[0, :]):
    #    r_j = np.sqrt(x[:, 0]**2 + yj**2) 
    #    ang_j = np.angle(z = x[:, 0] + 1j * yj)
    #    phi_j = np.empty_like(ang_j)
    #    for i in range(x[:, 0].shape[0]):
    #        phi_j[i] = ang_j[i] if (ang_j[i] > 0) else (ang_j[i] + 2*np.pi)


    #for i, xi in enumerate(x[:, 0]):
        #r_i = np.sqrt(xi**2 + y[0, :]**2)
        #angle_i = np.angle(z = xi + 1j * y[0, :])
        #phi_i = np.empty_like(angle_i)
        #for j in range(y[0, :].shape[0]):
        #    phi_i[j] = angle_i[j] if (angle_i[j] > 0) else (angle_i[j] + 2*np.pi)
        
    #for i, xi in enumerate(x[:, 0]):
        #for j, yj in enumerate(y[0, :]):
        #    r_ij = np.sqrt(xi**2 + yj**2)
        #    angle_ij = np.angle(z = xi + 1j * yj)
        #    phi_ij = angle_ij if (angle_ij > 0) else (angle_ij + 2*np.pi)
        #    if r_ij < 1:
        #        interp0 = np.copy(uD(r = r_ij, phi = phi_ij).evaluate()['g'])
        #        if rank == 0:
        #            interp = interp0
        #        else:
        #            interp = np.empty(2, dtype=np.float64)
        #        comm.Bcast(interp, root=0)
        #        print(rank, i, j, interp)
        #        #uC['g'][:, i, j] = interp[:, 0, 0]
        #        #uC['g'][0, i, j] *= 0.5 * (1 - np.tanh((r_ij - 0.9)/0.04))
        #        #uC['g'][1, i, j] *= 0.5 * (1 - np.tanh((r_ij - 0.9)/0.04))
        #        near_edge = 0.5 * (1 - np.tanh((r_ij - 0.9)/0.04))
        #        if (xi in xL) and (yj in yL): # if local
        #            uC['g'][:, i, j] = near_edge * interp
        #        else:
        #            uC['g']
#            else:
#                if (xi in xL) and (yj in yL):
#                    uC['g'][:, i, j] = 0.    
#                else:
#                    uC['g']
    # spectrum step
    logger.info("beginning spectrum step")
    spec = []
    for m in range(klow.shape[0]):
        # check whether we have at least one kh for this klow
        if num_mask[m]:
            Es = []
            for i, xi in enumerate(x[:, 0]):
                for j, yj in enumerate(y[0, :]): 
                    # only do calculation if in mask for this klow
                    if masks[m][i, j]:
                        E_m_ij = E(uC['c'][:, i, j], kh[i, j])
                        Es.append(E_m_ij)
                        print(i, j, E_m_ij)
            spec.append( np.sum(Es) / nums[m] / ((2*np.pi)**2) )
    spectra[w, :] = spec

# save
processed = {}
processed['nout'] = nwritesf
processed['khs'] = khs
processed['spectra'] = spectra

np.save('processed_spectra_' + output_suffix + '.npy', processed)

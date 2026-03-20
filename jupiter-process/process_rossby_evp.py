"""
Solve EVP to find Rossby modes associated with a given zonal flow profile

Usage:
    process_rossby_evp.py <file>... [options]

Options:
    --m=<int>           azimuthal wave number of Rossby modes to solve for [default: 1]
    --inviscid=<bool>   True: solves inviscid EVP with Dirichlet BC, False: solves dissipative EVP with Dirichlet+SF BC [default: True]
"""
import numpy as np
import h5py
import scipy.special as sp
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from pathlib import Path

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### read in options ###

from docopt import docopt
args = docopt(__doc__)
logger.info("args read in")
if rank == 0:
    print(args)
 
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

file_str = args['<file>'][0]
m = int(args['--m'])
inviscid = eval(args['--inviscid'])

output_prefix = 'processed_rossby_evp'
output_prefix += '_m_{:d}'.format(m)
output_prefix += '_inviscid_{:d}'.format(inviscid)

output_suffix = file_str.split('processed_profiles_')[1].split('.')[0].split('/')[0]
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
kf_str = output_suffix.split('kf_')[1].split('_')[0]

alpha_read = str_to_float(alpha_str)
gamma_read = str_to_float(gamma_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)
kf_read = str_to_float(kf_str)

alpha_vals = np.array((1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 1200, 1920, 2372, 2500, 3200))
eps_vals = np.array([1.0, 2.0])
nu_vals = np.array([2e-4, 8/90000, 8e-5, 4e-5, 2e-5])
kf_vals = np.array((10, 20, 30, 40, 80))

alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]
k_force = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]

### EVP setup ### 

# Method to accept/reject well-resolved modes based on Chp.7 of Boyd
def separate_resolved(evals_N_lo, evals_N_hi):
    
    # sort by absolute value
    idx_lo = np.abs(evals_N_lo).argsort()[::-1]
    evals_N_lo_sort = evals_N_lo[idx_lo]
    idx_hi = np.abs(evals_N_hi).argsort()[::-1]
    evals_N_hi_sort = evals_N_hi[idx_hi]

    len_lo, len_hi = (evals_N_lo_sort.shape[0], evals_N_hi_sort.shape[0])

    # calculate separations between adjacent (in abs val) modes from N_lo evals
    sigmas_lo = np.zeros(len_lo)
    sigmas_lo[0] = np.abs(evals_N_lo_sort[1] - evals_N_lo_sort[0])
    for i in range(len_lo-1):
        sigmas_lo[i] = 0.5*(np.abs(evals_N_lo_sort[i] - evals_N_lo_sort[i-1]) + np.abs(evals_N_lo_sort[i+1]-evals_N_lo_sort[i]))
    sigmas_lo[-1] = np.abs(evals_N_lo_sort[-2] - evals_N_lo_sort[-1]) 
    
    # calulate scaled differences between nearest evals from N_lo and N_hi
    idx_nearest = [np.argmin(np.abs(evals_N_lo_sort[i] - evals_N_hi_sort)/sigmas_lo[i]) for i in range(len_lo)]
    deltas = np.array([np.abs(evals_N_lo_sort[i] - evals_N_hi_sort[idx_nearest[i]])/sigmas_lo[i] for i in range(len_lo)]) 
    
    # apply threshold to inverse deltas
    drifts = 1/deltas
    drift_thresh = 1e3

    # separate 
    evals_resolved = evals_N_lo_sort[np.where(drifts > drift_thresh)[0]]
    evals_unresolved = evals_N_lo_sort[np.where(drifts <= drift_thresh)[0]]    
    idx_resolved = []
    idx_unresolved = []

    for eig in evals_resolved:
        idx_resolved.append(np.where(evals_N_lo_sort == eig))
    for eig in evals_unresolved:
        idx_unresolved.append(np.where(evals_N_lo_sort == eig))

    return evals_resolved, evals_unresolved, idx_resolved, idx_unresolved

# Dedalus EVP
def evp_dense(Nphi, Nr, u0filename, inviscid, params):
    
    if inviscid:
        m, gamma = params
    else:
        m, gamma, alpha, nu = params

    dtype = np.complex128
    dealias = 3/2
    coords = d3.PolarCoordinates('phi', 'r')
    dist = d3.Distributor(coords, dtype=dtype)
    disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dtype=dtype)
    edge = disk.edge
    radial_basis = disk.radial_basis
    phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

    # fields
    om = dist.Field(name='om')
    psi = dist.Field(name='psi', bases=disk)
    tau_psi = dist.Field(name='tau_psi')
    tau_psi2 = dist.Field(name='tau_psi2', bases=edge)
    lift = lambda A, n: d3.Lift(A, disk, n)
    if not inviscid:
        tau_psi3 = dist.Field(name='tau_psi3', bases=edge)
        er = dist.VectorField(coords, bases=radial_basis) #Vector er -- need to restrict to radial_basis?
        er['g'][1] = 1 
    f = dist.Field(name='f', bases = radial_basis)
    u0 = dist.VectorField(coords, name='u0', bases=radial_basis)
    
    # define backgrounds
    f.change_scales(dealias)
    f['g'] = - 0.5*gamma*pow(r_deal,2)
    
    u0file = np.load(u0filename, allow_pickle = True)[()] 
    um0_tavg = u0file['um0_tavg']
    Nr_in = um0_tavg.shape[0]
    Nphi_b = int(2*Nr_in/dealias)
    Nr_b = int(Nr_in/dealias)
    disk_in = d3.DiskBasis(coords, shape=(Nphi_b, Nr_b), radius=1, dtype=dtype)
    radial_basis_in = disk_in.radial_basis
    um0_in = dist.Field(bases=radial_basis_in)
    um0_in.change_scales(dealias)
    um0_in['g'] = np.copy(um0_tavg)
    
    u0.change_scales(dealias)
    um0_in.change_scales(dealias*Nr/Nr_b)
    u0['g'] = np.copy(um0_in['g'])
         
    dt = lambda A: 1j*om*A
    if not inviscid:
        dr = lambda A: er@d3.grad(A)
        sf = lambda A: dr(dr(A)) - dr(A)

    # EVP
    if inviscid:
        problem = d3.EVP([psi, tau_psi, tau_psi2], eigenvalue=om, namespace=locals())
        problem.add_equation("dt(lap(psi)) + u0@grad(lap(psi)) + skew(grad(psi))@grad(f) + lift(tau_psi2, -1) + tau_psi = 0")
    else:
        problem = d3.EVP([psi, tau_psi, tau_psi2, tau_psi3], eigenvalue=om, namespace=locals())
        problem.add_equation("dt(lap(psi)) + alpha*lap(psi) - nu*lap(lap(psi)) + u0@grad(lap(psi)) + skew(grad(psi))@grad(f) + lift(tau_psi2, -1) + lift(tau_psi3, -2) + tau_psi = 0")
        problem.add_equation("sf(psi)(r=1) = 0")
    problem.add_equation("psi(r=1) = 0")
    problem.add_equation("integ(psi) = 0")
    solver = problem.build_solver(ncc_cutoff=1e-6)
    sp = solver.subproblems_by_group[(m, None)]
    
    # Solve
    solver.solve_dense(sp)
    
    # Sort results
    idxs = np.abs(solver.eigenvalues).argsort()[::-1]
    evals = solver.eigenvalues[idxs]
    evecs = solver.eigenvectors[:,idxs]
    # throw out inf evals/evecs, if any
    evecs = evecs[:,np.logical_not(np.isinf(evals))]
    evals = evals[np.logical_not(np.isinf(evals))]
    
    return evals, evecs

### Solve EVPs and retain well-resolved modes###
dtype = np.complex128
dealias = 3/2
Nphi_lo, Nr_lo = (Nphi, Nr)
Nphi_hi, Nr_hi = (int(dealias*Nphi), int(dealias*Nr))

if inviscid:
    params = (m, gamma)
else:
    params = (m, gamma, alpha, nu)

# Solve at nominal resolution
evals_lo, evecs_lo = evp_dense(Nphi_lo, Nr_lo, file_str, inviscid, params)
# Solve at higher resolution
evals_hi, evecs_hi = evp_dense(Nphi_hi, Nr_hi, file_str, inviscid, params)

# Accept/reject
evals_resolved, evals_unresolved, idxs_resolved, idxs_unresolved = separate_resolved(evals_lo, evals_hi)

# Save end results
processed = {}
processed['evals_resolved'] = evals_resolved
processed['evecs_resolved'] = evecs_lo[:, idxs_resolved]
processed['evals_unresolved'] = evals_unresolved
processed['evecs_unresolved'] = evecs_lo[:, idxs_unresolved]
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

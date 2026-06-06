"""
Solve EVP to find Rossby modes associated with a given zonal flow profile

Usage:
    process_rossby_evp.py <file>... [options]

Options:
    --output=<str>      prefix in the name of the output file [default: processed_rossby_evp_nom1] 
    --m=<int>           azimuthal wave number of Rossby modes to solve for [default: 1]
    --inviscid=<bool>   True: solves inviscid EVP with Dirichlet BC, False: solves dissipative EVP with Dirichlet+SF BC [default: True]
    --save_psi=<bool>   [default: False]
    --save_vort=<bool>  [default: True]
    --save_bad=<bool>   [default: False] (only applies to evecs)
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
output = args['--output']
m = int(args['--m'])
inviscid = eval(args['--inviscid'])
save_psi = eval(args['--save_psi'])
save_vort = eval(args['--save_vort'])
save_bad = eval(args['--save_bad'])

output_prefix = output
output_prefix += '_m_{:d}'.format(m)
output_prefix += '_inviscid_{:d}'.format(inviscid)

output_suffix = file_str.split('processed_profiles_filtered_nom1_')[1].split('.')[0].split('/')[0]
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
# Adapted from eigentools/eigenproblem.py (Dedalus) 
def separate_resolved(evals_N_lo, evals_N_hi, thresh_opts):

    reverse_eval_lo_indx = np.arange(len(evals_N_lo)) 
    reverse_eval_hi_indx = np.arange(len(evals_N_hi))

    eval_lo_and_indx = np.asarray(list(zip(evals_N_lo, reverse_eval_lo_indx)))
    eval_hi_and_indx = np.asarray(list(zip(evals_N_hi, reverse_eval_hi_indx)))

    eval_lo_and_indx = eval_lo_and_indx[np.isfinite(evals_N_lo)]
    eval_hi_and_indx = eval_hi_and_indx[np.isfinite(evals_N_hi)]

    eval_lo_and_indx = eval_lo_and_indx[np.argsort(eval_lo_and_indx[:, 0].real)]
    eval_hi_and_indx = eval_hi_and_indx[np.argsort(eval_hi_and_indx[:, 0].real)]

    eval_lo_sorted = eval_lo_and_indx[:, 0]
    eval_hi_sorted = eval_hi_and_indx[:, 0]

    sigmas = np.zeros(len(eval_lo_sorted))
    sigmas[0] = np.abs(eval_lo_sorted[0] - eval_lo_sorted[1])
    sigmas[1:-1] = [0.5*(np.abs(eval_lo_sorted[j] - eval_lo_sorted[j - 1]) + np.abs(eval_lo_sorted[j + 1] - eval_lo_sorted[j])) for j in range(1, len(eval_lo_sorted) - 1)]
    sigmas[-1] = np.abs(eval_lo_sorted[-2] - eval_lo_sorted[-1])

    delta_near = np.array([np.nanmin(np.abs(eval_lo_sorted[j] - eval_hi_sorted)/sigmas[j]) for j in range(len(eval_lo_sorted))])

    inverse_drift = 1/delta_near

    # decision making
    drift_thresh, imag_thresh = thresh_opts

    drift_pass = inverse_drift > drift_thresh
    drift_fail = inverse_drift <= drift_thresh
    imag_pass = eval_lo_sorted.imag <= imag_thresh

    eval_lo_and_indx_res = eval_lo_and_indx[np.where(np.logical_or(drift_pass, imag_pass))]
    eval_lo_and_indx_bad = eval_lo_and_indx[np.where(drift_fail)]
    eval_lo_res = eval_lo_and_indx_res[:, 0]
    eval_lo_bad = eval_lo_and_indx_bad[:, 0]
    indx_res = eval_lo_and_indx_res[:, 1].real.astype(int)
    indx_bad = eval_lo_and_indx_bad[:, 1].real.astype(int)
    drifts_res = inverse_drift[np.where(np.logical_or(drift_pass, imag_pass))]    
    drifts_bad = inverse_drift[np.where(drift_fail)]

    return eval_lo_res, eval_lo_bad, indx_res, indx_bad, drifts_res, drifts_bad

# Dedalus EVP
def evp_dense(Nphi, Nr, u0filename, inviscid, params):
    print("Nphi", Nphi, "Nr", Nr)    

    if inviscid:
        m, gamma, save_psi, save_vort = params
    else:
        m, gamma, alpha, nu, save_psi, save_vort = params

    if (not save_psi) and (not save_vort):
        logger.info("Note: only eigenvalues will be returned")

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
    
    disk_in = d3.DiskBasis(coords, shape=(int(2*Nr_in), Nr_in), radius=1, dtype=dtype)
    radial_basis_in = disk_in.radial_basis
    um0_in = dist.Field(bases=radial_basis_in)
    um0_in.change_scales(1)
    um0_in['g'] = np.copy(um0_tavg)

    u0.change_scales(dealias)
    um0_in.change_scales(dealias*(Nr/Nr_in))
    u0['g'][0, :, :] = np.copy(um0_in['g'])
    u0['g'][1, :, :] *= 0.         

    dt = lambda A: 1j*om*A
    if not inviscid:
        #dr = lambda A: er@d3.grad(A)
        #sf = lambda A: dr(dr(A)) - dr(A)
        stress = lambda A: 0.5*(d3.grad(A) + d3.trans(d3.grad(A)))
        u_psi = lambda A: d3.skew(d3.grad(A))

    # EVP
    if inviscid:
        problem = d3.EVP([psi, tau_psi, tau_psi2], eigenvalue=om, namespace=locals())
        problem.add_equation("dt(lap(psi)) + u0@grad(lap(psi)) + skew(grad(psi))@grad(f) + lift(tau_psi2, -1) + tau_psi = 0")
    else:
        problem = d3.EVP([psi, tau_psi, tau_psi2, tau_psi3], eigenvalue=om, namespace=locals())
        problem.add_equation("dt(lap(psi)) + alpha*lap(psi) - nu*lap(lap(psi)) + u0@grad(lap(psi)) + skew(grad(psi))@grad(f) + lift(tau_psi2, -1) + lift(tau_psi3, -2) + tau_psi = 0")
        #problem.add_equation("sf(psi)(r=1) = 0")
        problem.add_equation("azimuthal(radial(stress(u_psi(psi))(r=1))) = 0")
    problem.add_equation("psi(r=1) = 0")
    problem.add_equation("integ(psi) = 0")
    solver = problem.build_solver(ncc_cutoff=1e-6)
    sp = solver.subproblems_by_group[(m, None)]

    # Solve
    logger.info("Initiating solve")
    #solver.solve_dense(sp)
    solver.solve_dense(sp, left=True, normalize_left=True)
    evals = solver.eigenvalues
    
    # Save fields
    logger.info("Saving results")
    psi_right_evecs = []
    vort_right_evecs = []
    psi_mleft_evecs = []
    vort_mleft_evecs = []
    if save_psi or save_vort:   
        for idx in range(evals.shape[0]):
            if idx % 25 == 0:
                logger.info("idx=%d, out of=%d" %(idx+1, evals.shape[0]))
            solver.set_state(idx, sp.subsystems[0])
            if save_psi:    
                psi.change_scales(dealias)
                psi_right_evecs.append(np.copy(psi['g']))
            if save_vort:
                vort = d3.lap(psi).evaluate()
                vort.change_scales(dealias)
                vort_right_evecs.append(np.copy(vort['g']))  
            solver.set_state(idx, sp.subsystems[0], modified_left=True) 
            if save_psi:    
                psi.change_scales(dealias)
                psi_mleft_evecs.append(np.copy(psi['g']))
            if save_vort:
                vort = d3.lap(psi).evaluate()
                vort.change_scales(dealias)
                vort_mleft_evecs.append(np.copy(vort['g'])) 
    
        if save_psi and save_vort:
            return evals, np.array(psi_right_evecs), np.array(psi_mleft_evecs), np.array(vort_right_evecs), np.array(vort_mleft_evecs)
        elif save_psi:
            return evals, np.array(psi_right_evecs), np.array(psi_mleft_evecs)
        elif save_vort:
            return evals, np.array(vort_right_evecs), np.array(vort_mleft_evecs)
        else:
            logger.info("This should never happen")
    else:
        logger.info("Returning evals")
        return evals

### Solve EVPs and retain well-resolved modes###
dtype = np.complex128
dealias = 3/2
Nphi_lo, Nr_lo = (Nphi, Nr)
Nphi_hi, Nr_hi = (int(dealias*Nphi), int(dealias*Nr))
print("lo", Nphi_lo, Nr_lo)
print("hi", Nphi_hi, Nr_hi)
if Nphi_hi % 2 != 0:
    Nphi_hi += 1
if Nr_hi % 2 != 0:
    Nr_hi += 1
print("hi", Nphi_hi, Nr_hi)

if inviscid:
    params = (m, gamma, save_psi, save_vort)
    params_hi = (m, gamma, False, False)
else:
    params = (m, gamma, alpha, nu, save_psi, save_vort)
    params_hi = (m, gamma, alpha, nu, False, False)

# Solve at nominal resolution
logger.info("Entering solve at nominal resolution")
if save_psi and save_vort:
    evals_lo, psi_right_evecs_lo, psi_mleft_evecs_lo, vort_right_evecs_lo, vort_mleft_evecs_lo = evp_dense(Nphi_lo, Nr_lo, file_str, inviscid, params)
elif save_psi:
    evals_lo, psi_right_evecs_lo, psi_mleft_evecs_lo = evp_dense(Nphi_lo, Nr_lo, file_str, inviscid, params)
elif save_vort:
    evals_lo, vort_right_evecs_lo, vort_mleft_evecs_lo = evp_dense(Nphi_lo, Nr_lo, file_str, inviscid, params)
else:
    evals_lo = evp_dense(Nphi_lo, Nr_lo, file_str, inviscid, params)

# Solve at higher resolution
logger.info("Entering solve at higher resolution")
evals_hi = evp_dense(Nphi_hi, Nr_hi, file_str, inviscid, params_hi)

# Accept/reject
drift_thresh = 1e3
imag_thresh = np.max((alpha * 3e2, 1e0))
thresh_opts = (drift_thresh, imag_thresh)
logger.info("Entering accept/reject procedure")

evals_lo_res, evals_lo_bad, indx_lo_res, indx_lo_bad, inverse_drift_res, inverse_drift_bad = separate_resolved(evals_lo, evals_hi, thresh_opts)
if save_psi:
    psi_right_evecs_lo_res = psi_right_evecs_lo[indx_lo_res, :]
    psi_right_evecs_lo_bad = psi_right_evecs_lo[indx_lo_bad, :]
    psi_mleft_evecs_lo_res = psi_mleft_evecs_lo[indx_lo_res, :]
    psi_mleft_evecs_lo_bad = psi_mleft_evecs_lo[indx_lo_bad, :]
if save_vort:
    vort_right_evecs_lo_res = vort_right_evecs_lo[indx_lo_res, :]
    vort_right_evecs_lo_bad = vort_right_evecs_lo[indx_lo_bad, :]
    vort_mleft_evecs_lo_res = vort_mleft_evecs_lo[indx_lo_res, :]
    vort_mleft_evecs_lo_bad = vort_mleft_evecs_lo[indx_lo_bad, :]

# Save end results
logger.info("Saving results as " + output_prefix + '_' + output_suffix + '.npy')
processed = {}
processed['evals_res'] = evals_lo_res
processed['evals_bad'] = evals_lo_bad
if save_psi:
    processed['psi_right_evecs_res'] = psi_right_evecs_lo_res
    if save_bad:
        processed['psi_right_evecs_bad'] = psi_right_evecs_lo_bad
    processed['psi_mleft_evecs_res'] = psi_mleft_evecs_lo_res
    if save_bad:
        processed['psi_mleft_evecs_bad'] = psi_mleft_evecs_lo_bad
if save_vort:
    processed['vort_right_evecs_res'] = vort_right_evecs_lo_res
    if save_bad:
        processed['vort_right_evecs_bad'] = vort_right_evecs_lo_bad
    processed['vort_mleft_evecs_res'] = vort_mleft_evecs_lo_res
    if save_bad:
        processed['vort_mleft_evecs_bad'] = vort_mleft_evecs_lo_bad
processed['drifts_res'] = inverse_drift_res
processed['drifts_bad'] = inverse_drift_bad


np.save(output_prefix + '_' + output_suffix + '.npy', processed)

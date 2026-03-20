"""
Bessel spectra -- this script is adjusted to return binned spectra for each m 

Usage:
    process_spectra_zb_mbin.py <file>... [options]

Options:
    --dini=<bool>               True: uses a Dini expansion (Bessel with H=1 Robin bc); False: uses a Bessel expansion (Dirichlet bc) [default: True]
    --t_out_start=<float>       Simulation time to begin making spectra [default: 0.]
    --t_out_end=<float>         Simulation time to stop making spectra [default: 100.]
    --t_steady_range=<float>    Size of time window prior to t_out_end to average over as "steady state" [default: 50.]
    --make_new=<bool>           Remake the Zernike to Bessel MMT matrices [default: False]
    --steady_only=<bool>        True: only saves the steady-state averaged spectral data; False: saves spectral data at all times for which it was calculated [default: True]
"""
import numpy as np
import h5py
import scipy.special as sp
from scipy.optimize import newton
import dedalus.public as d3
from mpi4py import MPI 
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from pathlib import Path

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
ring = int(output_suffix.split('ring_')[1].split('_')[0])

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

amp = np.sqrt(eps)

dini = eval(args['--dini'])
t_out_start = float(args['--t_out_start'])
t_out_end = float(args['--t_out_end'])
t_steady_range = float(args['--t_steady_range'])
make_new = eval(args['--make_new'])
steady_only = eval(args['--steady_only'])

output_prefix = 'processed_spectra_zb_mbin'
if dini: 
    output_prefix += '_dini'
else:
    output_prefix += '_std'
if steady_only:
    output_prefix += '_steady'

### Setup 
def robin_func(r, m, H): 
    return np.real(r * sp.jvp(m, r, n=1) + H * sp.jv(m, r)) 

def robin_func_prime(r, m, H): 
    return np.real(r * sp.jvp(m, r, n=2) + (H + 1) * sp.jvp(m, r, n=1))

def dini_roots(m, Nr, H):
    jmp_zs = sp.jnp_zeros(m, Nr + 1)
    djmp_zs = np.diff(jmp_zs)
    if m >= 1:
        r0 = sp.jnp_zeros(m, 1)
    else:
        r0 = [1.0]
    roots = []
    for nidx in range(Nr):
        rout, results = newton(robin_func, r0, fprime=robin_func_prime, args = (m, H), tol=1e-10, full_output=True)
        # catches
        if rout[0] <= 0:
            print("encountered a negative root")
            raise
        if rout[0] in np.unique(roots):
            print("caution: found a duplicate")
        if np.abs(robin_func(rout[0], m, H)) > 1e-10:
            print("caution: does not appear to be a root", np.abs(robin_func(rout[0], m, H)))
        roots.append(rout[0])
        r0 = rout + djmp_zs[nidx]
    return roots

def dini_weights_direct(m, dini_zsm):
    H = 1
    return ((H**2 + dini_zsm**2 - m**2) * sp.jv(m, dini_zsm)**2) / (2 * dini_zsm**2)

def dini_weights_psi_to_ke(m, dini_zsm):
    H = 1
    return ((H**2 - 2*H + dini_zsm**2 - m**2) * sp.jv(m, dini_zsm)**2) / (2 * dini_zsm**2)

def zern2dini(m, Nr, dini_zsm):
    nstart = int(np.floor(m/2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = (-1)**(jj - 1) * np.sqrt(2 * (2*jj + m - 1)) * (2 * dini_zsm[i] * sp.jv(2*jj + m - 1, dini_zsm[i])) / ((H**2 + dini_zsm[i]**2 - m**2) * sp.jv(m, dini_zsm[i])**2)
    return ZBm

def std_roots(m, Nr):
    return sp.jn_zeros(m, Nr)

def std_weights(m, std_zsm):
    return (sp.jv(m + 1, std_zsm)**2) / 2

def zern2std(m, Nr, std_zsm):
    nstart = int(np.floor(m/2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = (-1)**(jj - 1) * np.sqrt(2 * (2*jj + m - 1)) * (2 * sp.jv(2*jj + m - 1, std_zsm[i])) / (std_zsm[i] * sp.jv(m + 1, std_zsm[i])**2)
    return ZBm

#def ke_m_dini(m, Nr, uphimc, uphims, urmc, urms, dini_zs):
def ke_m_dini(m, Nr, upmc, upms, ummc, umms, dini_zs):
    dini_zsm = dini_zs[m, :]
    dini_weight_m = dini_weights_direct(m, dini_zsm)
    if m == 0:
        return 2 * np.pi * dini_weight_m * 0.5 * (upms**2 + umms**2) # counter-intuitive: nonzero m=0 Zernike coefficients are stored in these indices under the (unitary) transform to spin basis
    else:
        return np.pi * dini_weight_m * 0.5 * (upmc**2 + upms**2 + ummc**2 + umms**2)

def en_m_dini(m, Nr, vortmc, vortms, dini_zs):
    dini_zsm = dini_zs[m, :]
    dini_weight_m = dini_weights_direct(m, dini_zsm)
    if m == 0:
        return 2 * np.pi * dini_weight_m * vortmc**2
    else:
        return np.pi * dini_weight_m * (vortmc**2 + vortms**2)

def ke_m_std(m, Nr, psimc, psims, std_zs):
    std_zsm = std_zs[m, :]
    std_weight_m = std_weights(m, std_zsm)
    if m == 0:
        return 2 * np.pi * std_weight_m * 0.5 * std_zsm**2 * psimc**2
    else:
        return np.pi * std_weight_m * 0.5 * std_zsm**2 * (psimc**2 + psims**2)

def en_m_std(m, Nr, psimc, psims, std_zs):
    std_zsm = std_zs[m, :]
    std_weight_m = std_weights(m, std_zsm)
    if m == 0:
        return 2 * np.pi * std_weight_m * std_zsm**4 * psimc**2
    else:
        return np.pi * std_weight_m * std_zsm**4 * (psimc**2 + psims**2)

def lambda_k(ke_k, ks):
    return np.sqrt(np.cumsum(ks**2 * ke_k))

def flux_k(ke_k, lam_k, ks): # an estimate for the flux valid in the inverse cascade range only (Boffetta and Ecke 2012) 
    return np.pi**(-3/2) * lam_k * ks * ke_k

def zern2grid(m, Nr, r):
    nstart = int(np.floor(m/2))
    ZGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZGm[i, j] = r[i]**m * sp.eval_jacobi(jj - 1, 0, m, 2*r[i]**2 - 1) * np.sqrt(2*(2*jj + m - 1))
    return ZGm

def J2grid(m, Nr, r, zsm):
    BGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(Nr):
            BGm[i, j] = sp.jv(m, zsm[j] * r[i]) 
    return BGm

def makeZBs_dini(Nr, Nphi, dini_zs):
    mmax = int(Nphi/2) - 1
    ZBs = {}
    prog_cad = 10
    for m in range(mmax + 1):
        dini_zsm = dini_zs[m, :]
        ZBs[m] = zern2dini(m, Nr, dini_zsm)
        if m % prog_cad == 0:
            logger.info("makeZB loop: m = %d out of %d" %(m, mmax))
    return ZBs

def makeZBs_std(Nr, Nphi, std_zs):
    mmax = int(Nphi/2) - 1
    ZBs = {}
    prog_cad = 32
    for m in range(mmax + 1):
        std_zsm = std_zs[m, :]
        ZBs[m] = zern2std(m, Nr, std_zsm)
        if m % prog_cad == 0:
            logger.info("makeZB loop: m = %d out of %d" %(m, mmax))
    return ZBs

def makeZGs(Nr, Nphi, r):
    mmax = int(Nphi/2) - 1
    ZGs = {}
    prog_cad = 32
    for m in range(mmax + 1):
        ZGs[m] = zern2grid(m, Nr, r)
        if m % prog_cad == 0:
            logger.info("makeZG loop: m = %d out of %d" %(m, mmax))
    return ZGs

def makeBGs(Nr, Nphi, r, zs):
    mmax = int(Nphi/2) - 1
    BGs = {}
    prog_cad = 32
    for m in range(mmax + 1):
        zsm = zs[m, :]
        BGs[m] = J2grid(m, Nr, r, zsm)
        if m % prog_cad == 0:
            logger.info("makeBG loop: m = %d out of %d" %(m, mmax))
    return BGs

def m_map(m, Nphi): ### is this still correct for vector fields..?
    m_in = np.array(m)
    if not m_in.shape:
        m_in = np.array([m])
    m_out = 4 * m_in
    mask = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
    return m_out

def define_bins(zs, case):
    centers = zs[0, :]
    Nbins = centers.shape[0]
    edges = np.array([centers[ii] - 0.5*(centers[ii] - centers[ii - 1]) for ii in range(1, Nbins)])
    edges = np.concatenate(([0.], edges,  [centers[-1] + np.pi/2]))
    counts = []
    masks = {}
    for b in range(Nbins):
        mask = np.logical_and((zs <= edges[b+1]), (zs >= edges[b]))
        if case == 'zonal':
            mask[1:, :] = False
        elif case == 'non_zonal':
            mask[0, :] = False
        count = np.sum(mask)
        counts.append(count)
        masks[b] = mask
    return Nbins, centers, edges, counts, masks

def define_bins_m(zs, mmax, m_keep):
    centers = zs[0, :]
    Nbins = centers.shape[0]
    edges = np.array([centers[ii] - 0.5*(centers[ii] - centers[ii - 1]) for ii in range(1, Nbins)])
    edges = np.concatenate(([0.], edges,  [centers[-1] + np.pi/2]))
    counts = []
    masks = {}
    for b in range(Nbins):
        mask = np.logical_and((zs <= edges[b+1]), (zs >= edges[b]))
        for m in range(mmax + 1):
            if m != m_keep:
                mask[m, :] = False
        count = np.sum(mask)
        counts.append(count)
        masks[b] = mask
    return Nbins, centers, edges, counts, masks


def bin_spectra(data, widths, Nbins, masks, case):
    nspec = []
    for b in range(Nbins):
        mask = masks[b]
        width = widths[b]
        bin_sum = np.sum(data[mask] / width)
        nspec.append(bin_sum)
    return nspec

# Load in analysis data
f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

u = dist.VectorField(coords, name = 'u', bases = disk) # velocity
vort = dist.Field(name = 'vort', bases = disk) # scalar vertical vorticity
#uphi = dist.Field(name = 'uphi', bases = disk) # azimuthal component of velocity
#ur = dist.Field(name = 'ur', bases = disk) # radial component of velocity

# fields to solve for psi
if not dini:
    uphi = dist.Field(name = 'uphi', bases = disk) # azimuthal component of velocity
    psi = dist.Field(name = 'psi', bases = disk) # streamfunction
    rscal = dist.Field(bases=disk)
    rscal.change_scales(dealias)
    rscal['g'] = r_deal
    tau_psi = dist.Field(name='tau_psi', bases=edge)
    lift = lambda A: d3.Lift(A, disk, -1)
    tau_psi2 = dist.Field(name='tau_psi2')

##### Begin processing #####
mmax = int(Nphi/2) - 1
ms = np.arange(mmax + 1)
if dini:
    H = 1
    # get roots
    dini_zs = np.zeros((mmax + 1, Nr))
    for m in range(mmax + 1): 
        dini_zs[m, :] = dini_roots(m, Nr, H)
    # build matrices
    filedir = 'mmts_dini_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '/' 
    filename = 'zb_dini_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '.npy'
    if not make_new:
        ZBs = np.load(filedir + filename, allow_pickle=True)[()]
        logger.info('ZBs load successful')
    else:
        logger.info('ZBs not found, proceeding to make from scratch')
        ZBs = makeZBs_dini(Nr, Nphi, dini_zs)
        dir_path = Path(filedir)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(filedir + filename, ZBs, allow_pickle=True)
    logger.info('Finished accessing or building matrices')
else:
    # get roots
    std_zs = np.zeros((mmax + 1, Nr))
    for m in range(mmax + 1):
        std_zs[m, :] = std_roots(m, Nr)
    # build matrices
    filedir = 'mmts_J_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '/'
    filename = 'zb_J_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '.npy'
    if not make_new:
        ZBs = np.load(filedir + filename, allow_pickle=True)[()]
        logger.info('ZBs load successful')
    else:
        logger.info('ZBs not found, proceeding to make from scratch')
        ZBs = makeZBs_std(Nr, Nphi, std_zs)
        dir_path = Path(filedir)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(filedir + filename, ZBs, allow_pickle=True)
    logger.info('Finished accessing or building matrices')

# specify writes to make spectra of 
ws = np.arange(np.where(t <= t_out_start)[0][-1], np.where(t >= t_out_end)[0][0] + 1)
nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0

# Zernike coefficients
vortZ = np.zeros((nw, Nphi, Nr))
if dini:
    #uphiZ = np.zeros((nw, Nphi, Nr))
    #urZ = np.zeros((nw, Nphi, Nr))
    upZ = np.zeros((nw, Nphi, Nr))
    umZ = np.zeros((nw, Nphi, Nr))
else:
    psiZ = np.zeros((nw, Nphi, Nr))

# Bessel coefficients
vortB = np.zeros((nw, Nphi, Nr))
if dini:
    #uphiB = np.zeros((nw, Nphi, Nr))
    #urB = np.zeros((nw, Nphi, Nr))
    upB = np.zeros((nw, Nphi, Nr))
    umB = np.zeros((nw, Nphi, Nr))
else:
    psiB = np.zeros((nw, Nphi, Nr))

# 2d spectra
keB = np.zeros((nw, mmax + 1, Nr))
enB = np.zeros((nw, mmax + 1, Nr))

# radial spectra
# zonal and residual decompositon of radial spectra
masks_m = {}
if dini:
    Nbins, centers, edges, counts, masks = define_bins(dini_zs, None)
    for m in range(mmax + 1):
        _, _, _, _, mask_m = define_bins_m(dini_zs, mmax, m)
        masks_m[m] = mask_m
else:
    Nbins, centers, edges, counts, masks = define_bins(std_zs, None)
    for m in range(mmax + 1):
        _, _, _, _, mask_m = define_bins_m(std_zs, mmax, m)
        masks_m[m] = mask_m
bin_widths = np.diff(edges)

keBn = np.zeros((nw, Nbins))
enBn = np.zeros((nw, Nbins))
fluxBn = np.zeros((nw, Nbins))
keBmn = np.zeros((nw, mmax + 1, Nbins)) # similar to keB above, just grouped differently (and normalization?)
enBmn = np.zeros((nw, mmax + 1, Nbins))

### Loop over writes
prog_cad = 32
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        logger.info("Outer writes loop: i = %d out of %d" %(i, nw))

    ### Zernike coeffs
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)

    #if dini:
    #    u.change_scales(dealias)
    #    uphi.change_scales(dealias)
    #    ur.change_scales(dealias)
    #    uphi['g'] = np.copy(u['g'][0, :, :])
    #    ur['g'] = np.copy(u['g'][1, :, :])
    #    uphi.change_scales(1)
    #    ur.change_scales(1)

    # Solve for psi if needed
    if not dini:
        u.change_scales(dealias)
        uphi.change_scales(dealias)
        uphi['g'] = np.copy(u['g'][0, :, :])
        ruphi = rscal * uphi
        problem = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
        problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vort")
        problem.add_equation("psi(r=1) = 0")
        problem.add_equation("integ(psi) = -0.5 * integ(ruphi)")
        solver = problem.build_solver()
        solver.solve()
                
    vort.change_scales(1)
    vortZgather = comm.gather(np.copy(vort['c']), root = 0)
    if rank == 0:
        vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)
    
    if dini:
        u.change_scales(1)
        #uphiZgather = comm.gather(np.copy(u['c'][0, :, :])) # this is technically a misnomer... 
        #urZgather = comm.gather(np.copy(u['c'][1, :, :]))
        upZgather = comm.gather(np.copy(u['c'][0, :, :]))
        umZgather = comm.gather(np.copy(u['c'][1, :, :]))
        upZ[i, :, :] = np.array(upZgather).reshape(Nphi, Nr)
        umZ[i, :, :] = np.array(umZgather).reshape(Nphi, Nr)

        #uphiZgather = comm.gather(np.copy(uphi['c']))
        #urZgather = comm.gather(np.copy(ur['c']))
        #uphiZ[i, :, :] = np.array(uphiZgather).reshape(Nphi, Nr)
        #urZ[i, :, :] = np.array(urZgather).reshape(Nphi, Nr)

    else:
       psi.change_scales(1)
       psiZgather = comm.gather(np.copy(psi['c'][:, :]))
       psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)

    ### Zernike to Bessel
    if rank == 0:
        logger.info("Rank %d is mapping coefficients to Bessel space" %(rank))
        for m in range(mmax + 1):
            if m % prog_cad == 0:
                logger.info("main ZB loop: m = %d out of %d" %(m, mmax))
            ZB = ZBs[m]
            midx = m_map(m, Nphi)
            midxc, midxs = (midx, midx + 1)
    
            vortB[i, midxc, :] = (ZB @ vortZ[i, midxc, :][0, :]).reshape(1, Nr)
            vortB[i, midxs, :] = (ZB @ vortZ[i, midxs, :][0, :]).reshape(1, Nr)

            if dini:
                #uphiB[i, midxc, :] = (ZB @ uphiZ[i, midxc, :][0, :]).reshape(1, Nr)
                #uphiB[i, midxs, :] = (ZB @ uphiZ[i, midxs, :][0, :]).reshape(1, Nr)
                #urB[i, midxc, :] = (ZB @ urZ[i, midxc, :][0, :]).reshape(1, Nr)
                #urB[i, midxs, :] = (ZB @ urZ[i, midxs, :][0, :]).reshape(1, Nr)
                upB[i, midxc, :] = (ZB @ upZ[i, midxc, :][0, :]).reshape(1, Nr)
                upB[i, midxs, :] = (ZB @ upZ[i, midxs, :][0, :]).reshape(1, Nr)
                umB[i, midxc, :] = (ZB @ umZ[i, midxc, :][0, :]).reshape(1, Nr)
                umB[i, midxs, :] = (ZB @ umZ[i, midxs, :][0, :]).reshape(1, Nr)

                # 2d volume-integrated coefficients
                #keB[i, m, :] = ke_m_dini(m, Nr, uphiB[i, midxc, :], uphiB[i, midxs, :], urB[i, midxc, :], urB[i, midxs, :], dini_zs)
                keB[i, m, :] = ke_m_dini(m, Nr, upB[i, midxc, :], upB[i, midxs, :], umB[i, midxc, :], umB[i, midxs, :], dini_zs)
                enB[i, m, :] = en_m_dini(m, Nr, vortB[i, midxc, :], vortB[i, midxs, :], dini_zs)

            else:
                psiB[i, midxc, :] = (ZB @ psiZ[i, midxc, :][0, :]).reshape(1, Nr)
                psiB[i, midxs, :] = (ZB @ psiZ[i, midxs, :][0, :]).reshape(1, Nr)

                # 2d volume-integrated coefficients
                keB[i, m, :] = ke_m_std(m, Nr, psiB[i, midxc, :], psiB[i, midxs, :], std_zs)
                enB[i, m, :] = en_m_std(m, Nr, psiB[i, midxc, :], psiB[i, midxs, :], std_zs)

        for m in range(mmax + 1):
            # individual m, but arranged in same way
            keBmn[i, m, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks_m[m], None)
            if m < 3:
                print(keBmn[i, m, :])
            enBmn[i, m, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks_m[m], None)

        # spectra (summed over m, and normalized by bin widths)
        keBn[i, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks, None)
        enBn[i, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks, None)
        fluxBn[i, :] = flux_k(keBn[i, :], lambda_k(keBn[i, :], centers), centers)

if rank == 0:
    logger.info("Beginning final tasks on rank %d" %(rank))

    # Save outputs
    processed = {}
    
    processed['ws'] = ws
    processed['ts'] = tw
    processed['ms'] = ms
    if dini:
        processed['zs'] = dini_zs
    else:
        processed['zs'] = std_zs

    if not steady_only:
        # coeff
        processed['vortZ'] = vortZ
        processed['vortB'] = vortB
        if dini:
            #processed['uphiZ'] = uphiZ
            #processed['urZ'] = urZ
            #processed['uphiB'] = uphiB
            #processed['urB'] = urB
            processed['upZ'] = upZ
            processed['umZ'] = umZ
            processed['upB'] = upB
            processed['umB'] = umB
        else:
            processed['psiZ'] = psiZ
            processed['psiB'] = psiB

        # (m,n) spectra
        processed['keB'] = keB
        processed['enB'] = enB

        # integrated in m
        processed['keBn'] = keBn
        processed['enBn'] = enBn 
        processed['fluxBn'] = fluxBn

        processed['keBmn'] = keBmn
        processed['enBmn'] = enBmn

    processed['Nbins'] = Nbins
    processed['centers'] = centers
    processed['edges'] = edges
    processed['counts'] = counts
    processed['masks'] = masks
    processed['masks_m'] = masks_m

    # time-averaging
    if len(ws) > 1:
    
        tavg_end = tw[-1]
        tavg_start = tavg_end - t_steady_range
        tavg_end_idx = np.where(tw <= tavg_end)[0][-1]
        tavg_start_idx = np.where(tw >= tavg_start)[0][0]

        keB_tavg = np.mean(keB[tavg_start_idx:tavg_end_idx, :, :], axis = 0)
        enB_tavg = np.mean(enB[tavg_start_idx:tavg_end_idx, :, :], axis = 0)
        processed['keB_tavg'] = keB_tavg
        processed['enB_tavg'] = enB_tavg

        processed['keBsum_tavg'] = np.sum(keB_tavg, axis = 1)
        processed['enBsum_tavg'] = np.sum(enB_tavg, axis = 1)

        keBn_tavg = np.mean(keBn[tavg_start_idx:tavg_end_idx, :], axis = 0)
        enBn_tavg = np.mean(enBn[tavg_start_idx:tavg_end_idx, :], axis = 0)
        fluxBn_tavg = np.mean(fluxBn[tavg_start_idx:tavg_end_idx, :], axis = 0)
        processed['keBn_tavg'] = keBn_tavg
        processed['enBn_tavg'] = enBn_tavg
        processed['fluxBn_tavg'] = fluxBn_tavg
        keBmn_tavg = np.mean(keBmn[tavg_start_idx:tavg_end_idx, :, :], axis = 0)
        enBmn_tavg = np.mean(enBmn[tavg_start_idx:tavg_end_idx, :, :], axis = 0)
        processed['keBmn_tavg'] = keBmn_tavg
        processed['enBmn_tavg'] = enBmn_tavg

    logger.info("Saving on rank %d" %(rank))
    logger.info("Name: " + output_prefix + '_' + output_suffix + '.npy')
    np.save(output_prefix + '_' + output_suffix + '.npy', processed)
else:
    logger.info("Rank %d is done" %(rank))



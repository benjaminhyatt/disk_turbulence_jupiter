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


##### Options #####
make_new = False #--True if we need to remake the Bessel to Zernike MMT matrices
use_forcing = False #--True if we want to look at the spectrum after one time step due to forcing
coeff_not_grid = False #--True if we want to initialize the RHS of the psi bvp equation from coeff space
use_direct = True # get ke and en directly from vort, rather than psi

# Read in parameters
Nphi, Nr = 512, 256
nu = 2e-4
gamma = 0 #675 #1920
k_force = 20
alpha = 1e-2
amp = 1
ring = 0
restart_evolved = False #False #True
old = False #True #False
if old:
    eps = 2 * amp**2
else:
    eps = amp**2

##### Setup #####

def robin_func(r, m, H):
    return np.real(r * sp.jvp(m, r, n=1) + H * sp.jv(m, r))

def robin_func_prime(r, m, H):
    return np.real(r * sp.jvp(m, r, n=2) + (H + 1) * sp.jvp(m, r, n=1))

def dini_roots(m, Nr, H):
    jmp_zs = sp.jnp_zeros(m, Nr + 1)
    djmp_zs = np.diff(jmp_zs)
    r0 = sp.jnp_zeros(m, 1)
    roots = []
    for nidx in range(Nr):
        rout, results = newton(robin_func, r0, fprime=robin_func_prime, args = (m, H), tol=1e-10, full_output=True)
        # catches 
        if rout[0] <= 0:
            print("encountered a negative root")
            raise
        if rout[0] in np.unique(roots):
            print("caution: found a duplicate")
        roots.append(rout[0])
        r0 = rout + djmp_zs[nidx]
    return roots

def dini_weights(m, dini_zsm):
    H = 1
    return ((H**2 + dini_zsm**2 - m**2) * sp.jv(m, dini_zsm)**2) / (2 * dini_zsm**2)

def J_roots(m, Nr):
    return sp.jn_zeros(m, Nr)

def zern2dini(m, Nr, dini_zsm):
    #H = 1
    #dini_zs = dini_roots(m, Nr, H)
    nstart = int(np.floor(m/2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = (-1)**(jj - 1) * np.sqrt(2 * (2*jj + m - 1)) * (2 * dini_zsm[i] * sp.jv(2*jj + m - 1, dini_zsm[i])) / ((H**2 + dini_zsm[i]**2 - m**2) * sp.jv(m, dini_zsm[i])**2)
    return ZBm

def zern2grid(m, Nr, r):
    nstart = int(np.floor(m/2))
    ZGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZGm[i, j] = r[i]**m * sp.eval_jacobi(jj - 1, 0, m, 2*r[i]**2 - 1) * np.sqrt(2*(2*jj + m - 1))
    return ZGm

def dini2grid(m, Nr, r, dini_zs):
    #H = 1 
    dini_zsm = dini_zs[m, :] #= dini_roots(m, Nr, H)
    BGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(Nr):
            BGm[i, j] = sp.jv(m, dini_zsm[j] * r[i]) 
    return BGm

def makeZBs(Nr, Nphi, dini_zs):
    mmax = int(Nphi/2) - 1
    ZBs = {}
    prog_cad = 10
    for m in range(mmax + 1):
        dini_zsm = dini_zs[m, :]
        ZBs[m] = zern2dini(m, Nr, dini_zsm)
        if m % prog_cad == 0:
            logger.info("makeZB loop: m = %d out of %d" %(m, mmax))
    return ZBs

def makeZGs(Nr, Nphi, r):
    mmax = int(Nphi/2) - 1
    ZGs = {}
    prog_cad = 10
    for m in range(mmax + 1):
        ZGs[m] = zern2grid(m, Nr, r)
        if m % prog_cad == 0:
            logger.info("makeZG loop: m = %d out of %d" %(m, mmax))
    return ZGs

def makeBGs(Nr, Nphi, r):
    mmax = int(Nphi/2) - 1
    BGs = {}
    prog_cad = 10
    for m in range(mmax + 1):
        BGs[m] = dini2grid(m, Nr, r)
        if m % prog_cad == 0:
            logger.info("makeBG loop: m = %d out of %d" %(m, mmax))
    return BGs

def m_map(m, Nphi):
    m_in = np.array(m)
    if not m_in.shape:
        m_in = np.array([m])
    m_out = 4 * m_in
    mask = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
    return m_out

def psi2vort(m, Nr, psim, dini_zs):
    #H = 1
    jzsm = dini_zs[m, :] #dini_roots(m, Nr, H)
    return - jzsm**2 * psim

def ke_m(m, Nr, psimc, psims, dini_zs):
    jzsm = dini_zs[m, :]
    dini_weight_m = dini_weights(m, jzsm)
    return 0.5 * jzsm * dini_weight_m * (np.abs(psimc) + np.abs(psims))

def en_m(m, Nr, psimc, psims, dini_zs):
    jzsm = dini_zs[m, :] 
    dini_weight_m = dini_weights(m, jzsm)
    return jzsm**2 * dini_weight_m * (np.abs(psimc) + np.abs(psims))

def vort2psi(m, Nr, vortm, dini_zs):
    jzsm = dini_zs[m, :]
    return vortm / (- jzsm**2)

def ke_m_direct(m, Nr, vortmc, vortms, dini_zs):
    jzsm = dini_zs[m, :]
    dini_weight_m = dini_weights(m, jzsm)
    psimc = vort2psi(m, Nr, vortmc, dini_zs)
    psims = vort2psi(m, Nr, vortms, dini_zs) 
    return ke_m(m, Nr, psimc, psims, dini_zs)

def en_m_direct(m, Nr, vortmc, vortms, dini_zs):
    jzsm = dini_zs[m, :]
    dini_weight_m = dini_weights(m, jzsm)
    psimc = vort2psi(m, Nr, vortmc, dini_zs)
    psims = vort2psi(m, Nr, vortms, dini_zs)
    return en_m(m, Nr, psimc, psims, dini_zs)

def define_bins(dini_zs):
    Nbins = np.ceil((dini_zs.max() - dini_zs.min()) / np.pi).astype(int)
    centers = dini_zs.min() + np.pi * np.arange(Nbins)
    edges = centers - np.pi/2
    edges = np.concatenate((edges, [centers[-1] + np.pi/2]))
    counts = [] # histogram
    masks = {}
    for b in range(Nbins):
        mask = np.logical_and((dini_zs <= edges[b+1]), (dini_zs >= edges[b]))
        count = np.sum(mask)
        counts.append(count)
        masks[b] = mask
    return Nbins, centers, edges, counts, masks

def bin_spectra(data, Nbins, counts, masks):
    nspec = []
    for b in range(Nbins):
        mask = masks[b]
        weighted_sum = np.sum(data[mask]) / counts[b]
        nspec.append(weighted_sum)
    return nspec

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
#output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) 
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

# Load in analysis data
#f = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
f = h5py.File('/anvil/projects/x-mth250004/jupiter/' + output_suffix + '/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
t = np.array(f['tasks/u'].dims[0]['sim_time'])

dealias = 3/2 
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype, comm = MPI.COMM_SELF)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

coords_cart = d3.CartesianCoordinates('x', 'y')
dist_cart = d3.Distributor(coords_cart, dtype=dtype, comm = MPI.COMM_SELF)
xbasis = d3.RealFourier(coords_cart['x'], size=Nr, bounds=(-1, 1), dealias=dealias)
ybasis = d3.RealFourier(coords_cart['y'], size=Nr, bounds=(-1, 1), dealias=dealias)

x_cart, y_cart = dist_cart.local_grids(xbasis, ybasis)
xm, ym = np.meshgrid(x_cart, y_cart)

kx = xbasis.wavenumbers[::2]
ky = np.concatenate( (ybasis.wavenumbers[::2], - ybasis.wavenumbers[2::2]))
kx, ky = np.meshgrid(kx,ky)
k = np.sqrt(kx**2 + ky**2) / np.pi

mask = (k >= k_force) & (k < k_force+1)
x_disk = r_deal*np.cos(phi_deal)
y_disk = r_deal*np.sin(phi_deal)

transform = 2*np.exp(1j*kx[mask][None,:]*x_disk.flatten()[:,None]
                   + 1j*ky[mask][None,:]*y_disk.flatten()[:,None])
phi_2d = (phi_deal + 0*r_deal).ravel()
transform_vector_rotated = np.stack( (1j*ky[mask][None,:]*transform, -1j*kx[mask][None,:]*transform ) ) / k_force**2
transform_vector = np.stack( (-transform_vector_rotated[0]*np.sin(phi_2d[:,None]) + transform_vector_rotated[1]*np.cos(phi_2d[:,None]),
                               transform_vector_rotated[0]*np.cos(phi_2d[:,None]) + transform_vector_rotated[1]*np.sin(phi_2d[:,None])))

k_len = len(k[mask])
phi_len = len(phi_deal.ravel())
r_len = len(r_deal.ravel())
transform_vector = transform_vector.reshape(2, phi_len, r_len, k_len)

rand = np.random.RandomState(seed=10001)
phases = rand.uniform(0,2*np.pi,k_len)
forcing = np.real(transform_vector @ np.exp(1j*phases))

rvec= dist.VectorField(coords, bases=disk)
rvec['g'][1] = r
ephi = dist.VectorField(coords, bases=disk) #Vector ephi
ephi['g'][0] = 1

rscal = dist.Field(bases=disk)
rscal['g'] = r

v1=dist.VectorField(coords, bases=disk)
v1.preset_scales(dealias)

u = dist.VectorField(coords, name='u', bases=disk)
p = dist.Field(name='p', bases=disk)
tau_u = dist.VectorField(coords, name='tau_u', bases=edge)
tau_p = dist.Field(name='tau_p')

stress = 0.5*(d3.grad(u) + d3.trans(d3.grad(u)))
lift = lambda A: d3.Lift(A, disk, -1) 
lift_2 = lambda A: d3.Lift(A, radial_basis, -2) 
sig = -np.sqrt(Nr/(Nr-1))

def forcing_func():
    phases = rand.uniform(0,2*np.pi,k_len)
    f1=np.real(transform_vector @ np.exp(1j*phases))
    if ring:
        f1 *= 0.5*(1-np.tanh( (r_deal-0.75)/width ))
    v1['g'] = f1
    angf1=d3.integ(rvec@d3.skew(v1)).evaluate()

    phases2 = rand.uniform(0,2*np.pi,k_len)
    f2=np.real(transform_vector @ np.exp(1j*phases2))
    if ring:
        f2 *= 0.5*(1-np.tanh( (r_deal-0.75)/width ))
    v1['g'] = f2
    angf2=d3.integ(rvec@d3.skew(v1)).evaluate()

    #if rank == 0:
    #    data = [angf1['g'][0][0]]*size
    #else:
    #    data = None
    #angf1_int= MPI.COMM_WORLD.scatter(data, root=0)
    angf1_int = angf1['g'][0][0]
    #if rank == 0:
    #    data = [angf2['g'][0][0]]*size
    #else:
    #    data = None
    #angf2_int= MPI.COMM_WORLD.scatter(data, root=0)
    angf2_int = angf2['g'][0][0]

    f=f1/angf1_int - f2/angf2_int

    v1['g'] = f
    norm=d3.integ(v1@v1).evaluate()
    #if rank == 0:
    #    data = [norm['g'][0][0]]*size
    #else:
    #    data = None
    #norm_int= MPI.COMM_WORLD.scatter(data, root=0)
    norm_int = norm['g'][0][0]
    return f * np.sqrt(2/norm_int/tstep)
F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)

problemF = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problemF.add_equation("div(u) + tau_p = 0")
problemF.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u")
problemF.add_equation("radial(u(r=1)) = 0", condition='nphi!=0')
problemF.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi!=0')
problemF.add_equation("radial(u(r=1)) = 0", condition='nphi==0')
problemF.equations[-1]['valid_modes'][1] = True
problemF.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi==0')
problemF.equations[-1]['valid_modes'][1] = True
problemF.add_equation("integ(p) = 0")

tstep = 1e-5
timestepper = d3.SBDF2
if use_forcing:
    logger.info("Building IVP")
    solverF = problemF.build_solver(timestepper)

##### Begin processing #####
mmax = int(Nphi/2) - 1
ms = np.arange(mmax + 1)

H = 1
jzs_dini = np.zeros((mmax + 1, Nr))
for m in range(mmax + 1):
    jzs_dini[m, :] = dini_roots(m, Nr, H)

# Build matrices
filedir = 'mmts_dini_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '/'
filename = 'zb_dini_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '.npy'
if not make_new:
    ZBs = np.load(filedir + filename, allow_pickle=True)[()]
    logger.info('ZBs load successful')
else:
    logger.info('ZBs not found, proceeding to make from scratch')
    #ZBs = makeZBs(Nr, Nphi)
    ZBs = makeZBs(Nr, Nphi, jzs_dini)
    dir_path = Path(filedir)
    dir_path.mkdir(parents=True, exist_ok=True)
    np.save(filedir + filename, ZBs, allow_pickle=True)
    
# Define writes to make spectra of 
if use_forcing:
    ws = [1]
else:
    t_out_start = 0.
    t_out_end = 200.
    ws = np.arange(np.where(t <= t_out_start)[0][-1], np.where(t >= t_out_end)[0][0] + 1)

    #t_per_w = 0.02
    #t_out_start = 0.
    #t_out_end = 299.
    #ws = np.arange(int(t_out_start/t_per_w), int(t_out_end/t_per_w) + 1)
    #ws = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
    #ws = np.linspace(700, 900, 21).astype(int)

nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0

# fields
vort = dist.Field(name = 'vort', bases = disk) # scalar vertical vorticity
psi = dist.Field(name = 'psi', bases = disk) # streamfunction
tau_psi = dist.Field(name='tau_psi', bases=edge)
lift = lambda A: d3.Lift(A, disk, -1)
tau_psi2 = dist.Field(name='tau_psi2')

# Perform Z2B transforms

vortZ = np.zeros((nw, Nphi, Nr))
psiZ = np.zeros((nw, Nphi, Nr))

vortB = np.zeros((nw, Nphi, Nr))
psi2vortB = np.zeros((nw, Nphi, Nr))
psiB = np.zeros((nw, Nphi, Nr))

keB = np.zeros((nw, int(Nphi/2), Nr))
enB = np.zeros((nw, int(Nphi/2), Nr))

Nbins, centers, edges, counts, masks = define_bins(jzs_dini)

keBn = np.zeros((nw, Nbins))
enBn = np.zeros((nw, Nbins))

prog_cad = 10
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        logger.info("Outer writes loop: i = %d out of %d" %(i, nw))

    if use_forcing:
        # for testing: obtain psi via the forcing (or just take a single time step? 2?)      
        logger.info("Taking steps")
        solverF.step(tstep)
        vortF = -d3.div(d3.skew(u)).evaluate()
        uphi = ephi@u
        ruphi = rscal * uphi

        vortrhs = dist.Field(bases = disk)
        if coeff_not_grid:
            vortrhs['c'] = np.copy(vortF['c'])
            
            # saving
            vortZ[i, :, :] =np.array(np.copy(vortrhs['c']))
            #vortZgather = comm.gather(np.copy(vortrhs['c']), root = 0)
            #if rank == 0:
            #    vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)

            logger.info("Solving for psi")
            problem = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
            problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vortrhs")
            problem.add_equation("psi(r=1) = 0")
            problem.add_equation("integ(psi) = - integ(ruphi)")
            solver = problem.build_solver()
            solver.solve()

            # saving
            psi.change_scales(1)
            psiZ[i, :, :] = np.array(np.copy(psi['c']))
            #psiZgather = comm.gather(np.copy(psi['c']), root = 0)
            #if rank == 0:
            #    psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)

        else:
            vortrhs.change_scales(dealias)    
            vortrhs['g'] = np.copy(vortF['g'])
            vortrhs.change_scales(1)
            
            # saving
            vortZ[i, :, :] = np.array(np.copy(vortrhs['c']))
            #vortZgather = comm.gather(np.copy(vortrhs['c']), root = 0)
            #if rank == 0:
            #    vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)

            logger.info("Solving for psi")
            problem = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
            problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vortF")
            problem.add_equation("psi(r=1) = 0")
            problem.add_equation("integ(psi) = - integ(ruphi)")
            solver = problem.build_solver()
            solver.solve()

            # saving
            psi.change_scales(1)
            psiZ[i, :, :] = np.array(np.copy(psi['c']))
            #psiZgather = comm.gather(np.copy(psi['c']), root = 0)
            #if rank == 0:
            #    psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)

    else:
        # load vorticity data and solve for psi
        vort.load_from_hdf5(f, w)
        
        # saving
        vortZ[i, :, :] = np.array(np.copy(vort['c']))
        #vortZgather = comm.gather(np.copy(vort['c']), root = 0)
        #if rank == 0:
        #    vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)

        # also need u to make correct gauge choice
        u.load_from_hdf5(f, w)
        uphi = ephi@u
        ruphi = rscal * uphi

        logger.info("Solving for psi")
        problem = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
        problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vort")
        problem.add_equation("psi(r=1) = 0")
        problem.add_equation("integ(psi) = - integ(ruphi)")
        solver = problem.build_solver()
        solver.solve()        
        
        # saving
        psi.change_scales(1)
        psiZ[i, :, :] = np.array(np.copy(psi['c']))
        #psiZgather = comm.gather(np.copy(psi['c']), root = 0)
        #if rank == 0:
        #        psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)

    # map psi and vort to Bessel (Dini)
    if rank == 0:
        logger.info("Rank %d is mapping coefficients to Bessel (Dini) space" %(rank))
        for m in range(mmax + 1):
            if m % prog_cad == 0:
                logger.info("main ZB loop: m = %d out of %d" %(m, mmax))
            ZB = ZBs[m]
            midx = m_map(m, Nphi)
            midxc, midxs = (midx, midx + 1)
     
            psiBc = (ZB @ psiZ[i, midxc, :][0, :]).reshape(1, Nr)
            psiBs = (ZB @ psiZ[i, midxs, :][0, :]).reshape(1, Nr)
            psiB[i, midxc, :] = psiBc
            psiB[i, midxs, :] = psiBs
            psi2vortB[i, midxc, :] = psi2vort(m, Nr, psiBc, jzs_dini)
            psi2vortB[i, midxs, :] = psi2vort(m, Nr, psiBs, jzs_dini)

            vortBc = (ZB @ vortZ[i, midxc, :][0, :]).reshape(1, Nr) 
            vortBs = (ZB @ vortZ[i, midxs, :][0, :]).reshape(1, Nr) 
            vortB[i, midxc, :] = vortBc
            vortB[i, midxs, :] = vortBs
                
            if use_direct:
                keB[i, m, :] = ke_m_direct(m, Nr, vortBc, vortBs, jzs_dini)
                enB[i, m, :] = en_m_direct(m, Nr, vortBc, vortBs, jzs_dini)
            else:
                keB[i, m, :] = ke_m(m, Nr, psiBc, psiBs, jzs_dini)
                enB[i, m, :] = en_m(m, Nr, psiBc, psiBs, jzs_dini)


        # binning to integrate over m
        keBn[i, :] = bin_spectra(keB[i, :, :], Nbins, counts, masks)
        enBn[i, :] = bin_spectra(enB[i, :, :], Nbins, counts, masks)

if rank == 0:
    logger.info("Beginning final tasks on rank %d" %(rank))

    # Save outputs
    processed = {}
    
    processed['ws'] = ws
    processed['ts'] = tw
    processed['ms'] = ms
    processed['jzs_dini'] = jzs_dini

    processed['vortZ'] = vortZ
    processed['psiZ'] = psiZ
    processed['vortB'] = vortB # direct transform from vortZ to Bessel (Dini)
    processed['psi2vortB'] = psi2vortB # Laplacian of psiB
    processed['psiB'] = psiB # direct transform of psiZ to Bessel (Dini)

    processed['keB'] = keB
    processed['enB'] = enB

    processed['keBsum'] = np.sum(keB, axis = 1)
    processed['enBsum'] = np.sum(enB, axis = 1)

    processed['keBn'] = keBn
    processed['enBn'] = enBn 

    processed['phi'] = phi
    processed['r'] = r

    processed['Nbins'] = Nbins
    processed['centers'] = centers
    processed['edges'] = edges
    processed['counts'] = counts
    processed['masks'] = masks

    # Time averaging
    if len(ws) > 1:
        processed['vortB_tavg'] = np.mean(vortB, axis = 0) 
        processed['psi2vortB_tavg'] = np.mean(psi2vortB, axis = 0)
        processed['psiB_tavg'] = np.mean(psiB, axis = 0)
        
        keB_tavg = np.mean(keB, axis = 0)
        enB_tavg = np.mean(enB, axis = 0)
        processed['keB_tavg'] = keB_tavg
        processed['enB_tavg'] = enB_tavg

        processed['keBsum_tavg'] = np.sum(keB_tavg, axis = 1)
        processed['enBsum_tavg'] = np.sum(enB_tavg, axis = 1)

        keBn_tavg = np.mean(keBn, axis = 0)
        enBn_tavg = np.mean(enBn, axis = 0)
        processed['keBn_tavg'] = keBn_tavg
        processed['enBn_tavg'] = enBn_tavg

    logger.info("Saving on rank %d" %(rank))

    if use_forcing:
        if coeff_not_grid:
            opt_str = 'c'
        else:
            opt_str = 'g'
        np.save('processed_spectra_zb_dini_' + output_suffix + '_F_' + opt_str + '.npy', processed)
    else:
        np.save('processed_spectra_zb_dini_' + output_suffix + '.npy', processed)
else:
    logger.info("Rank %d is done" %(rank))



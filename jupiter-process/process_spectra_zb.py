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


##### Options #####
make_new = False
use_forcing = True #False #True #--True if we want to look at the spectrum after one time step due to forcing

# Read in parameters
Nphi, Nr =  512, 256
nu = 2e-4
gamma = 0
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

def J_roots(m, Nr):
    return sp.jn_zeros(m, Nr)

def zern2bess(m, Nr):
    J_zs = J_roots(m, Nr)
    nstart = int(np.floor(m/2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = (-1)**(jj - 1) * np.sqrt(2 * (2*jj + m - 1)) * (2 * sp.jv(2*jj + m - 1, J_zs[i])) / (J_zs[i] * sp.jv(m + 1, J_zs[i])**2)
    return ZBm

def zern2grid(m, Nr, r):
    nstart = int(np.floor(m/2))
    ZGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZGm[i, j] = r[i]**m * sp.eval_jacobi(jj - 1, 0, m, 2*r[i]**2 - 1) * np.sqrt(2*(2*jj + m - 1))
    return ZGm

def bess2grid(m, Nr, r):
    J_zs = J_roots(m, Nr)
    BGm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(Nr):
            BGm[i, j] = sp.jv(m, J_zs[j] * r[i]) 
    return BGm

def makeZBs(Nr, Nphi):
    mmax = int(Nphi/2) - 1
    ZBs = {}
    prog_cad = 10
    for m in range(mmax + 1):
        ZBs[m] = zern2bess(m, Nr)
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
        BGs[m] = bess2grid(m, Nr, r)
        if m % prog_cad == 0:
            logger.info("makeBG loop: m = %d out of %d" %(m, mmax))
    return BGs

def psi2ke(m, Nr, psim):
    J_zs = J_roots(m, Nr)
    return 0.5 * J_zs * psim

def psi2enstrophy(m, Nr, psim):
    J_zs = J_roots(m, Nr)
    return J_zs**2 * psim

def psi2vort(m, Nr, psim):
    J_zs = J_roots(m, Nr)
    return - J_zs**2 * psim

def m_map(m, Nphi):
    m_in = np.array(m)
    if not m_in.shape:
        m_in = np.array([m])
    m_out = 4 * m_in
    mask = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
    return m_out

#def psi2ke(m, Nr, psim):
#    J_zs = J_roots(m, Nr)
#    kem = np.zeros(Nr)
#    for i in range(Nr):
#        kem[i] = 0.5 * (J_zs[i] * psim[i])**2
#    return kem

#def psi2enstrophy(m, Nr, psim):
#    J_zs = J_roots(m, Nr)
#    enstrophym = np.zeros(Nr)
#    for i in range(Nr):
#        enstrophym[i] = (J_zs[i]**2 * psim[i])**2
#    return enstrophym

#def m_map(m_in, Nphi):
#    msnew = []
#    for m in m_in:
#        midx = 4 * m
#        if midx > Nphi - 2:
#            midx = Nphi - 2 - 4 * (m - int(Nphi/4))
#        msnew.append(midx)
#    return np.array(msnew)

#def psi2ke(m, Nr, psimc, psims):
#    J_zs = J_roots(m, Nr)
#    return 0.5 * J_zs**2 * (psimc**2 + psims**2)

#def psi2enstrophy(m, Nr, psims, psimc):
#    J_zs = J_roots(m, Nr)
#    return J_zs[i]**4 * (psimc**2 + psims**2)

#def m_map(m, Nphi):
#    m_in = np.array(m)
#    if not m_in.shape:
#        m_in = np.array([m])
#    m_out = 4 * m_in
#    mask = m_out > Nphi - 2
#    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
#    return m_out
        
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
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

coords_cart = d3.CartesianCoordinates('x', 'y')
dist_cart = d3.Distributor(coords_cart, dtype=dtype)
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

    if rank == 0:
        data = [angf1['g'][0][0]]*size
    else:
        data = None
    angf1_int= MPI.COMM_WORLD.scatter(data, root=0)
    if rank == 0:
        data = [angf2['g'][0][0]]*size
    else:
        data = None
    angf2_int= MPI.COMM_WORLD.scatter(data, root=0)

    f=f1/angf1_int - f2/angf2_int

    v1['g'] = f
    norm=d3.integ(v1@v1).evaluate()
    if rank == 0:
        data = [norm['g'][0][0]]*size
    else:
        data = None
    norm_int= MPI.COMM_WORLD.scatter(data, root=0)

    return f * np.sqrt(2/norm_int/tstep)
F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)

problemF = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problemF.add_equation("div(u) + tau_p = 0")
problemF.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u")
problemF.add_equation("radial(u(r=1)) = 0", condition='nphi!=0')
problemF.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi!=0')
problemF.add_equation("radial(u(r=1)) = 0", condition='nphi==0')
#try:
problemF.equations[-1]['valid_modes'][1] = True
#except:
#    logger.info("Skipping valid modes line on rank %d" %(rank))
problemF.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi==0')
#try:
problemF.equations[-1]['valid_modes'][1] = True
#except:
#logger.info("Skipping valid modes line on rank %d" %(rank))
problemF.add_equation("integ(p) = 0")

tstep = 1e-5
timestepper = d3.SBDF2
logger.info("Building IVP")
solverF = problemF.build_solver(timestepper)

#solverF.step(tstep)
#vortF = -d3.div(d3.skew(u)).evaluate()
#print(vortF['c'][0, :])

##### Begin processing #####

# Build matrices
filedir = 'mmts_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '/'
filename = 'zb_Nr_{:}'.format(Nr) + '_Nphi_{:}'.format(Nphi) + '.npy'
if not make_new:
    ZBs = np.load(filedir + filename, allow_pickle=True)[()]
    logger.info('ZBs load successful')
else:
    logger.info('ZBs not found, proceeding to make from scratch')
    ZBs = makeZBs(Nr, Nphi)
    dir_path = Path(filedir)
    dir_path.mkdir(parents=True, exist_ok=True)
    np.save(filedir + filename, ZBs, allow_pickle=True)

### Processing ### 
mmax = int(Nphi/2) - 1
ms = np.arange(mmax + 1)

# Select writes to make spectra of 
if use_forcing:
    ws = [1]
else:
    ws = [1, 1000, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0

# fields
vort = dist.Field(name = 'vort', bases = disk) # scalar vertical vorticity
psi = dist.Field(name = 'psi', bases = disk) # streamfunction
tau_psi = dist.Field(name='tau_psi', bases=edge)
lift = lambda A: d3.Lift(A, disk, -1)
tau_psi2 = dist.Field(name='tau_psi2')

# Perform Z2B transforms
psiZ = np.zeros((nw, Nphi, Nr))
psiB = np.zeros((nw, Nphi, Nr))
keB = np.zeros((nw, Nphi, Nr)) #np.zeros((nw, int(Nphi/2), Nr)) #np.zeros((nw, Nphi, Nr))
enstrophyB = np.zeros((nw, Nphi, Nr)) #np.zeros((nw, int(Nphi/2), Nr)) #np.zeros((nw, Nphi, Nr))
#temporary
vortB = np.zeros((nw, Nphi, Nr))
keZ = np.zeros((nw, Nphi, Nr))
#keG = np.zeros((nw, Nphi, Nr))
enZ = np.zeros((nw, Nphi, Nr))
vortZ = np.zeros((nw, Nphi, Nr))

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
        #vortm0F = d3.Average(vortF, coords['phi']) # * 2 * np.pi
        #print(vortF['c'][0, :])

        # part2
        #vortc = dist.Field(bases = disk)
        #vortc['c'] = np.copy(vortF['c'])


        ### the below gave back what we expected, now I am just throwing in one more thing
        #vortF.change_scales(1)
        #print(vortF['c'][0, :])

        vortg = dist.Field(bases = disk)
        vortg.change_scales(dealias)
        vortg['g'] = vortF['g']
        vortg.change_scales(1)
        vortZ = np.copy(vortg['c'])
        
        logger.info("Solving for psi")
        #problem = d3.LBVP([psi, tau_psi], namespace=locals())
        problem = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
        problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vortF")
        #problem.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vortc")
        problem.add_equation("psi(r=1) = 0")
        problem.add_equation("integ(psi) = - integ(ruphi)")
        #problem.add_equation("integ(psi) = - integ(uphi)")
        #problem.add_equation("integ(psi) = 0")

        #problem.add_equation("lap(psi) + lift(tau_psi) = vortF", condition="nphi!=0")
        #problem.add_equation("psi(r=1) = 0", condition="nphi!=0")
        #problem.add_equation("lap(psi) + lift(tau_psi) = vortF", condition="nphi==0")
        ###problem.add_equation("lap(psi) + lift(tau_psi) = vortm0F", condition="nphi==0")
        #problem.add_equation("integ(psi) = 0", condition="nphi==0")
        ###problem.add_equation("integ(psi) = - integ(uphi)", condition="nphi==0")

        #problem.add_equation("lap(psi) + lift(tau_psi) = vortF", condition="(nphi!=0)and(nr!=0)")
        #problem.add_equation("psi(r=1) = 0", condition="(nphi!=0)and(nr!=0)")
        #problem.add_equation("psi = 0", condition="(nphi==0)and(nr==0)")
        #problem.add_equation("tau_psi = 0", condition="(nphi==0)and(nr==0)")

        #problem.add_equation("lap(psi) + lift(tau_psi) = vortF", condition="nphi!=0")
        #problem.add_equation("psi(r=1) = 0", condition="nphi!=0")
        #problem.add_equation("psi = -uphi", condition="nphi==0")
        #problem.add_equation("tau_psi = 0", condition="nphi==0")

        #problem.add_equation("lap(psi) + lift(tau_psi) = vortF", condition="nphi!=0")
        #problem.add_equation("psi(r=1) = 0", condition="nphi!=0")
        #problem.add_equation("psi = vortm0F", condition="nphi==0")
        #problem.add_equation("tau_psi = 0", condition="nphi==0")

        solver = problem.build_solver()
        solver.solve() 

        #print(vortF['c'][0, :])

    else:
        # load vorticity data and solve for psi
        vort.load_from_hdf5(f, w)
        vortm0 = d3.Average(vort, coords['phi'])
        # also need u to make correct gauge choice
        u.load_from_hdf5(f, w)
        uphi = ephi@u

        logger.info("Solving for psi")
        problem = d3.LBVP([psi, tau_psi], namespace=locals())
        problem.add_equation("lap(psi) + lift(tau_psi) = vort", condition="nphi!=0")
        problem.add_equation("psi(r=1) = 0", condition="nphi!=0")
        problem.add_equation("lap(psi) + lift(tau_psi) = vortm0", condition="nphi==0")
        problem.add_equation("integ(psi) = -integ(uphi)", condition="nphi==0") 
        
        #problem.add_equation("lap(psi) + lift(tau_psi) = vort", condition="nphi!=0")
        #problem.add_equation("psi(r=1) = 0", condition="nphi!=0")
        #problem.add_equation("psi = vortm0", condition="nphi==0")
        #problem.add_equation("tau_psi = 0", condition="nphi==0")

        solver = problem.build_solver()
        solver.solve()

    # temporary:
    ke = (0.5 * u@u).evaluate()
    ke.change_scales(1)
    keZgather = comm.gather(ke['c'], root = 0)
    if use_forcing:
        vort = vortF
    #print(vortF['c'].shape)
    #vortZgather = comm.gather(vortF['c'], root = 0)
    #if rank == 0:
        #vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)
        #print(vortZ[i, 0, :])
    en = (vort * vort).evaluate()
    en.change_scales(1)
    enZgather = comm.gather(en['c'], root = 0)
    if rank == 0:
        keZ[i, :, :] = np.array(keZgather).reshape(Nphi, Nr) 
        enZ[i, :, :] = np.array(enZgather).reshape(Nphi, Nr) 

    #keGgather = comm.gather(ke['g'], root = 0)
    #if rank == 0:
    #    keZ[i, :, :] = np.array(keZgather).reshape(Nphi, Nr) 
    #    keG[i, :, :] = np.array(keGgather).reshape(Nphi, Nr)

    # optional: save psi in Zernike basis
    #psi.change_scales(1)
    #print(psi['c'].shape)
    
    #psig = dist.Field(bases = disk)
    #psig['g'] = psi['g']
    #lappsi = d3.lap(psig).evaluate()['g']
    #psic = dist.Field(bases = disk)
    #psic['c'] = np.copy(psi['c'])
    #psiZG = np.copy(psic['g'])
    #lappsiZG = np.copy(d3.lap(psic).evaluate()['g'])
    

    psiZgather = comm.gather(psi['c'], root = 0)
    if rank == 0:
        psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)
    else:
        logger.info("Rank %d is done with write = %d" %(rank, w))

    # map psi to Bessel basis
    if rank == 0:
        logger.info("Rank %d is mapping coefficients to Bessel space" %(rank))
        for m in range(mmax + 1):
            if m % prog_cad == 0:
                logger.info("main ZB loop: m = %d out of %d" %(m, mmax))
            ZB = ZBs[m]
            midx = m_map(m, Nphi)
            midxc, midxs = (midx, midx + 1)
     
            #if m < 20:
            #    print(m, ZB)
            #midx = 4 * m
            #if midx > Nphi - 2:
            #    midx = Nphi - 2 - 4 * (m - int(Nphi/4))
            #midxc, midxs = (midx, midx + 1)
            psiBc = (ZB @ psiZ[i, midxc, :][0, :]).reshape(1, Nr) #psi['c'][midxc, :]
            psiBs = (ZB @ psiZ[i, midxs, :][0, :]).reshape(1, Nr) #psi['c'][midxs, :]
            #if m < 4:
                #print(m, psiZ[i, midxc, :])
                #print(ZB.shape, psiZ[i, midxc, :].shape, psiBc.shape)
            
            # retain same organization for now
            psiB[i, midxc, :] = psiBc
            psiB[i, midxs, :] = psiBs
            vortB[i, midxc, :] = psi2vort(m, Nr, psiBc)
            vortB[i, midxs, :] = psi2vort(m, Nr, psiBs)

            keB[i, midxc, :] = psi2ke(m, Nr, psiBc)
            keB[i, midxs, :] = psi2ke(m, Nr, psiBs)
            enstrophyB[i, midxc, :] = psi2enstrophy(m, Nr, psiBc)
            enstrophyB[i, midxs, :] = psi2enstrophy(m, Nr, psiBs)

            # combine cos and sin modes, and make order nice
            #keB[i, m, :] = psi2ke(m, Nr, psiBc, psiBs)
            #enstrophyB[i, m, :] = psi2enstrophy(m, Nr, psiBc, psiBs)

if rank == 0:
    logger.info("Beginning final tasks on rank %d" %(rank))
    jzs = np.zeros((Nphi, Nr))
    for m in range(mmax + 1):
        jzs[m, :] = J_roots(m, Nr)

    # Save outputs
    processed = {}
    processed['ws'] = ws
    processed['ts'] = tw
    processed['ms'] = ms
    processed['jzs'] = jzs
    processed['psiZ'] = psiZ
    processed['psiB'] = psiB
    processed['keB'] = keB # Zernike coefficients can be access from analysis files later if needed
    processed['enstrophyB'] = enstrophyB # Zernike coefficients can be access from analysis files later if needed
    # temporary
    processed['keZ'] = keZ
    #processed['keG'] = keG
    processed['enZ'] = enZ
    processed['phi'] = phi
    processed['r'] = r
    processed['vortZ'] = vortZ
    processed['vortB'] = vortB

    #processed['lappsi'] = lappsi
    #processed['psiZG'] = psiZG
    #processed['lappsiZG'] = lappsiZG


    # Time averaging
    if len(ws) > 1:
        processed['psiZ_tavg'] = np.mean(psiZ, axis = 0)
        processed['psiB_tavg'] = np.mean(psiB, axis = 0)
        processed['keB_tavg'] = np.mean(keB, axis = 0)
        processed['enstrophyB_tavg'] = np.mean(enstrophyB, axis = 0)

    logger.info("Saving on rank %d" %(rank))
    np.save('processed_spectra_zb_' + output_suffix + '.npy', processed)
else:
    logger.info("Rank %d is done" %(rank))



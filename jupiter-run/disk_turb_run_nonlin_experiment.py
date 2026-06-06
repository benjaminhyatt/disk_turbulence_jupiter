import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import scipy

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.Get_size()

from dedalus.tools.logging import *
logger = logging.getLogger(__name__)

# Parameters
dealias = 3/2
dtype = np.float64

seed_in = 31415926
alpha = 1e-2
gamma = 400
eps = 1
amp = np.sqrt(eps)
nu = 2e-4
k_force = 20
logger.info("k_force, adjusted: %d" %(k_force))

Nphi = 512
Nr = 256

ring = 0
width = 0.04 

ncc_cutoff = 1e-6

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis

phi, r = dist.local_grids(disk, scales=(1, 1))
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

# forcing
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
#k = np.sqrt(kx**2 + ky**2) / (2 * np.pi)

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

rand = np.random.RandomState(seed=seed_in) #10001
phases = rand.uniform(0,2*np.pi,k_len)
forcing = np.real(transform_vector @ np.exp(1j*phases))

# Fields
u = dist.VectorField(coords, name='u', bases=disk)
p = dist.Field(name='p', bases=disk)
tau_u = dist.VectorField(coords, name='tau_u', bases=edge)
tau_p = dist.Field(name='tau_p')

stress = 0.5*(d3.grad(u) + d3.trans(d3.grad(u)))
lift = lambda A: d3.Lift(A, disk, -1)
lift_2 = lambda A: d3.Lift(A, radial_basis, -2)
sig = -np.sqrt(Nr/(Nr-1))

er = dist.VectorField(coords, bases=disk) #Vector er
er['g'][1] = 1
ephi = dist.VectorField(coords, bases=disk) #Vector ephi
ephi['g'][0] = 1
rvec= dist.VectorField(coords, bases=disk)
rvec['g'][1] = r

vort = -d3.div(d3.skew(u))
angm = -rvec@d3.skew(u)

v1=dist.VectorField(coords, bases=disk)
v1.preset_scales(dealias)

# Define GeneralFunction for forcing
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
#F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)

G = dist.Field(bases=disk)
G.preset_scales(dealias)
G['g']= - 0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) #Coriolis parameter
G=d3.Grid(G)
pvort = vort + G

# Background
u0 = dist.VectorField(coords, bases=disk)
u0.change_scales(dealias)
u0['g'][0] = 3 * r_deal

# Initial condition (wave -- for now just a simple first guess, rather than an EVP output)

psi = dist.Field(bases=disk)
psi.change_scales(dealias)
k11 = scipy.special.jn_zeros(1, 1)[-1]
psi['g'] = scipy.special.jv(1, k11 * r_deal) * np.cos(phi_deal)

u.change_scales(dealias)
u_psi = lambda A: d3.skew(d3.grad(A))
u['g'] = u_psi(psi)['g']
u['g'] += u0['g']

# second wave
psi2 = dist.Field(bases=disk)
psi2.change_scales(dealias)
k12 = scipy.special.jn_zeros(1, 2)[-1]
psi2['g'] = scipy.special.jv(1, k12 * r_deal) * np.cos(phi_deal)

u['g'] += u_psi(psi2)['g']

# Problem
problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - alpha*u - G*d3.skew(u) -u@grad(u)")
problem.add_equation("radial(u(r=1)) = 0", condition='nphi!=0')
problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi!=0')
problem.add_equation("radial(u(r=1)) = 0", condition='nphi==0')
try:
    problem.equations[-1]['valid_modes'][1] = True
except:
    logger.info("Skipping valid modes line on rank %d" %(rank))
problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi==0')
try:
    problem.equations[-1]['valid_modes'][1] = True
except:
    logger.info("Skipping valid modes line on rank %d" %(rank))
problem.add_equation("integ(p) = 0")

# timestepping
stop_time = 5
timestepper = d3.SBDF2
tstep = 1e-5

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper, ncc_cutoff=ncc_cutoff)
solver.stop_sim_time = stop_time
logger.info('solver built')

# Analysis
analysis = solver.evaluator.add_file_handler('analysis_nonlin_with_m0_2', sim_dt = 0.005, mode='overwrite')
vort0 = -d3.div(d3.skew(u0))
analysis.add_task(vort - vort0, layout='g', name='vort')
analysis.add_task(u, layout='g', name='u')

vort_field = dist.Field(bases=disk, name='vort_field')
analysis.add_task(-d3.div(d3.skew(u)), layout='g', name='vort_field')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(tstep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, tstep, max_u))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

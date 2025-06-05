import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import os.path
import sys

##  $ mpiexec -n 4 python3 disk_turb.py --restart
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

from dedalus.tools.logging import *
logger = logging.getLogger(__name__)

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.Get_size()

# Parameters
dealias = 3/2
dtype = np.float64
alpha = 1e-2
amp = 1
Nphi, Nr = 1024, 512
nu = 5e-5
gamma = 240 #30
k_force = 50
#Nphi, Nr = 512, 256
#nu = 2e-4 # did 2e-3 by accident..
#gamma = 240 #30
#k_force = 20
ring = 0 #1
width = 0.08

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.0e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
output_suffix = output_suffix.replace('-','m').replace('+','p')

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis

phi, r = dist.local_grids(disk, scales=(dealias, dealias))
phi_deal, r_deal = dist.local_grids(disk, scales=(1, 1)) #need this size for the force

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

k = np.sqrt(kx**2 + ky**2)/np.pi

mask = (k >= k_force) & (k < k_force+1)
x_disk = r*np.cos(phi)
y_disk = r*np.sin(phi)

transform = 2*np.exp(1j*kx[mask][None,:]*x_disk.flatten()[:,None]
                   + 1j*ky[mask][None,:]*y_disk.flatten()[:,None])
phi_2d = (phi + 0*r).ravel()
transform_vector_rotated = np.stack( (1j*ky[mask][None,:]*transform, -1j*kx[mask][None,:]*transform ) ) / k_force**2
transform_vector = np.stack( (-transform_vector_rotated[0]*np.sin(phi_2d[:,None]) + transform_vector_rotated[1]*np.cos(phi_2d[:,None]), 
                               transform_vector_rotated[0]*np.cos(phi_2d[:,None]) + transform_vector_rotated[1]*np.sin(phi_2d[:,None])))

k_len = len(k[mask])
phi_len = len(phi.ravel())
r_len = len(r.ravel())
transform_vector = transform_vector.reshape(2, phi_len, r_len, k_len)

rand = np.random.RandomState(seed=10001)
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
rvec['g'][1] = r_deal

vort = -d3.div(d3.skew(u))
angm = rvec@d3.skew(u)

v1=dist.VectorField(coords, bases=disk)
v1.preset_scales(dealias)

# Define GeneralFunction for forcing
def forcing_func():
    phases = rand.uniform(0,2*np.pi,k_len)
    f1=np.real(transform_vector @ np.exp(1j*phases))
    if ring:
        f1 *= 0.5*(1-np.tanh( (r-0.75)/width ))
    v1['g'] = f1
    angf1=d3.integ(rvec@d3.skew(v1)).evaluate()
 
    phases2 = rand.uniform(0,2*np.pi,k_len)
    f2=np.real(transform_vector @ np.exp(1j*phases2))
    if ring:
        f2 *= 0.5*(1-np.tanh( (r-0.75)/width ))
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
        
    return f *2 /np.sqrt(norm_int) /np.sqrt(tstep)      #ensures amp = epsilon  

F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)
angF= rvec@d3.skew(F)
Fvort=d3.div(d3.skew(F))

G = dist.Field(bases=disk)
G['g']= - 0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) #Coriolis parameter
G=d3.Grid(G)

# Problem
problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
if restart:
    problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
else:
    problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
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
stop_time = 10/alpha
timestepper = d3.SBDF2
tstep = 5e-5

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_time
logger.info('solver built')

# Restart from checkpoint
if restart:
    write, initial_timestep = solver.load_state('checkpoints_' + output_suffix +'/checkpoints_' + output_suffix + '_s20.h5')
    file_handler_mode = 'append'
    rand = np.random.RandomState(seed=10001+solver.iteration)
else:
    file_handler_mode = 'overwrite'

# Analysis
analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = 0.1, mode=file_handler_mode)
# scalars
analysis.add_task(d3.integ(vort), name = 'W')
analysis.add_task(d3.integ(angm), name = 'Lzu')
analysis.add_task(d3.integ(Fvort), name = 'FW')
analysis.add_task(d3.integ(angF), name = 'LzF')
analysis.add_task(d3.integ(0.5*u@u), name = 'KE')
analysis.add_task(d3.integ(vort*vort), name = 'EN')
# profiles
vortm0 = d3.Average(vort, coords['phi'])
analysis.add_task(vortm0, name = 'w0')
drvortm0 = d3.Average(er@d3.grad(vort), coords['phi'])
analysis.add_task(drvortm0, name='drw0')
# snapshots
analysis.add_task(vort, layout='g', name='vorticity')
analysis.add_task(u, layout='g', name='u') # so we can estimate energy spectrum in post

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

CFL = d3.CFL(solver, initial_dt=tstep, cadence=10, safety=0.15, threshold=0.05, max_dt=1e-4)
CFL.add_velocity(u)

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints_' + output_suffix, sim_dt = 10, max_writes = 1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        tstep = CFL.compute_timestep()
        solver.step(tstep)
        if (solver.iteration-1) % 100 == 0:
            max_u = np.sqrt(flow.max('u2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e" %(solver.iteration, solver.sim_time, tstep, max_u))
            if max_u > 1e4 or np.isnan(max_u):
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

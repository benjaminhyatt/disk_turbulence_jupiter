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

logger.info("this script uses a width of 0.025 for the ring tanh option")

# Parameters
#Nphi, Nr = 256, 128
Nphi, Nr = 512, 256
dealias = 3/2
dtype = np.float64
nu = 2e-4 # 0.3e-4
alpha = 1e-2
amp = -1
gamma = 0
k_force = 20 # 50
ring=1 
width = 0.25

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
#ky = ybasis.wavenumbers[::2]
ky = np.concatenate( (ybasis.wavenumbers[::2], - ybasis.wavenumbers[2::2]))
kx, ky = np.meshgrid(kx,ky)

k = np.sqrt(kx**2 + ky**2)/np.pi

mask = (k >= k_force) & (k < k_force+1)
x_disk = r*np.cos(phi)
y_disk = r*np.sin(phi)

transform = 2*np.exp(1j*kx[mask][None,:]*x_disk.flatten()[:,None]
                   + 1j*ky[mask][None,:]*y_disk.flatten()[:,None])

#transform_vector = np.stack( (1j*ky[mask][None,:]*transform, -1j*kx[mask][None,:]*transform ) ) / k_force**2

phi_2d = (phi + 0*r).ravel()
transform_vector_rotated = np.stack( (1j*ky[mask][None,:]*transform, -1j*kx[mask][None,:]*transform ) ) / k_force**2
transform_vector = np.stack( (-transform_vector_rotated[0]*np.sin(phi_2d[:,None]) + transform_vector_rotated[1]*np.cos(phi_2d[:,None]), 
                               transform_vector_rotated[0]*np.cos(phi_2d[:,None]) + transform_vector_rotated[1]*np.sin(phi_2d[:,None])))


k_len = len(k[mask])
phi_len = len(phi.ravel())
r_len = len(r.ravel())
transform_vector = transform_vector.reshape(2, phi_len, r_len, k_len)

rand = np.random.RandomState(seed=49492)
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

rvec= dist.VectorField(coords, bases=disk) #Vector er
rvec['g'][1] = r_deal

ur=er@u
uphi=ephi@u

tau_phir = stress@ephi@er
tau_rphi = stress@er@ephi
tau_rr = stress@er@er
t_rphi_R = ephi@(er@stress(r=1))  

vort = -d3.div(d3.skew(u))
vort2 = vort*vort
angm= rvec@d3.skew(u)

duphi=er@d3.grad(uphi) #dr(uphi)
duphi2= er@d3.grad(duphi) # dr(dr(uphi))
duphi2_R=duphi2(r=1) # dr(dr(uphi)) at edge (function of phi)

lapvort=d3.lap(vort)

v1=dist.VectorField(coords, bases=disk)
v1.preset_scales(dealias)

# Define GeneralFunction for forcing
def forcing_func():
    phases = rand.uniform(0,2*np.pi,k_len)
    f1=np.real(transform_vector @ np.exp(1j*phases))
    if ring:
        #f1 *= 0.5*(1-np.tanh( (r-0.75)/0.08 ))
        f1 *= 0.5*(1-np.tanh( (r-0.75)/width ))
    v1['g'] = f1
    curlf1=d3.integ(d3.div(d3.skew(v1))).evaluate()
    
    phases2 = rand.uniform(0,2*np.pi,k_len)
    f2=np.real(transform_vector @ np.exp(1j*phases2))
    if ring:
        #f2 *= 0.5*(1-np.tanh( (r-0.75)/0.08 ))
        f2 *= 0.5*(1-np.tanh( (r-0.75)/width ))
    v1['g'] = f2
    curlf2=d3.integ(d3.div(d3.skew(v1))).evaluate()

    # v1 is randomized phase data transformed to grid and multiplied by tanh function on grid 
    # curlf1 and curlf2 are volume-integrated curls of the vector field v1

    if rank == 0:
        data = [curlf1['g'][0][0]]*size
    else:
        data = None
    curlf1_int= MPI.COMM_WORLD.scatter(data, root=0) 
    if rank == 0:
        data = [curlf2['g'][0][0]]*size
    else:
        data = None
    curlf2_int= MPI.COMM_WORLD.scatter(data, root=0) 
    
    # if average curl of either is sufficiently close to zero, then use that f
    # otherwise, construct a third f that will
    if (np.abs(curlf1_int) <1e-8):
       return f1
    if (np.abs(curlf2_int) < 1e-8):
       return f2
    else: 
       f=f1/curlf1_int - f2/curlf2_int
       v1['g'] = f
       norm=d3.integ(v1@v1).evaluate()
       if rank == 0:
           data = [norm['g'][0][0]]*size
       else:
           data = None
       norm_int= MPI.COMM_WORLD.scatter(data, root=0)
        
       return f *2 /np.sqrt(norm_int) /np.sqrt(tstep)      #ensures amp = epsilon  
    return f1

F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)
curlF=d3.integ(d3.div(d3.skew(F)))
angF= rvec@d3.skew(F)
Fvort=d3.div(d3.skew(F))
Fr=er@F 
curlF_R=Fvort(r=1)

G = dist.Field(bases=disk)

#Gop = dist.Field(bases=disk)
#Gop=  0.5*gamma**pow(r,2) * pow(phi,0) 
#G['g'] = Gop.evaluate()  #rhs is not dealiased, so not use G['g']in lhs?
G['g']= - 0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) #Coriolis parameter
#uskew.evaluate()
G=d3.Grid(G)

# For debug: auxilary problem fields
wz = dist.Field(name='wz', bases=disk)
wz_2 = dist.Field(name='wz_2', bases=disk)
wz_lap = dist.Field(name='wz_lap', bases=disk)
wz_tau = dist.Field(name='wz_tau', bases=disk)
wz_tau_2 = dist.Field(name='wz_tau_2', bases=disk)
wz_adv = dist.Field(name='wz_adv', bases=disk)
wz_for = dist.Field(name='wz_for', bases=disk)
wz_dam = dist.Field(name='wz_dam', bases=disk)

# Problem
problem = d3.IVP([p, u, tau_u, tau_p, wz, wz_2, wz_lap, wz_tau, wz_tau_2, wz_adv, wz_for, wz_dam], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
# modified: 
problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u")
problem.add_equation("radial(u(r=1)) = 0", condition='nphi!=0')
problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi!=0')
problem.add_equation("radial(u(r=1)) = 0", condition='nphi==0')
if not problem.equations[-1]['valid_modes'][1]:
    problem.equations[-1]['valid_modes'][1] = True
else:
    logger.info("Skipping valid modes line on rank %d" %(rank))
problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi==0')
if not problem.equations[-1]['valid_modes'][1]:
    problem.equations[-1]['valid_modes'][1] = True
else:
    logger.info("Skipping valid modes line on rank %d" %(rank))
problem.add_equation("integ(p) = 0")
# auxilary
# default:
problem.add_equation("dt(wz) + nu*div(skew(lap(u))) - div(skew(lift(tau_u))) = div(skew(u@grad(u))) - amp*div(skew(F)) + alpha*div(skew(u))")
# modified:
problem.add_equation("dt(wz_2) + nu*div(skew(lap(u))) - div(skew(lift(tau_u))) - sig*div(skew(lift_2(tau_u))) = div(skew(u@grad(u))) - amp*div(skew(F)) + alpha*div(skew(u))")
problem.add_equation("dt(wz_lap) + nu*div(skew(lap(u))) = 0")
problem.add_equation("dt(wz_tau) - div(skew(lift(tau_u))) = 0")
#problem.add_equation("dt(wz_tau_2) - div(skew(lift(tau_u))) - sig*div(skew(lift_2(tau_u))) = 0")
problem.add_equation("dt(wz_tau_2) - sig*div(skew(lift_2(tau_u))) = 0")
problem.add_equation("dt(wz_adv) = div(skew(u@grad(u)))")
problem.add_equation("dt(wz_for) = -amp*div(skew(F))")
problem.add_equation("dt(wz_dam) = alpha*div(skew(u))")

# timestepping
#stop_iteration = 100000000
stop_time = 500
timestepper = d3.SBDF2
tstep = 5e-5

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper)
#solver.stop_iteration = stop_iteration
solver.stop_sim_time = stop_time
logger.info('solver built')

ephi_tau = dist.VectorField(coords, bases=edge)
ephi_tau['g'][0] = 1 
er_tau = dist.VectorField(coords, bases=edge)
er_tau['g'][1] = 1

# Analysis

# auxilary equation analysis tasks
analysis_aux = solver.evaluator.add_file_handler('analysis_debug_aux_changeF', sim_dt = 0.1)
analysis_aux.add_task(d3.integ(wz_lap), name='W_lap')
analysis_aux.add_task(d3.integ(wz_tau), name='W_tau')
analysis_aux.add_task(d3.integ(wz_tau_2), name='W_tau_2')
analysis_aux.add_task(d3.integ(wz_adv), name='W_adv')
analysis_aux.add_task(d3.integ(wz_for), name='W_for')
analysis_aux.add_task(d3.integ(wz_dam), name='W_dam')
lap_rate = -d3.div(d3.skew(nu*d3.lap(u)))
analysis_aux.add_task(d3.integ(lap_rate), name='lap_rate')
tau_rate = d3.div(d3.skew(lift(tau_u)))
analysis_aux.add_task(d3.integ(tau_rate), name='tau_rate')
tau_rate_2 = sig*d3.div(d3.skew(lift_2(tau_u)))
analysis_aux.add_task(d3.integ(tau_rate_2), name='tau_rate_2')
adv_rate = -d3.div(d3.skew(-d3.DotProduct(u, d3.grad(u))))
analysis_aux.add_task(d3.integ(adv_rate), name='adv_rate')
for_rate = -d3.div(d3.skew(amp*F))
analysis_aux.add_task(d3.integ(for_rate), name='for_rate')
dam_rate = d3.div(d3.skew(alpha*u))
analysis_aux.add_task(d3.integ(dam_rate), name='dam_rate')

analysis_aux.add_task(d3.integ(wz), name='wz')
analysis_aux.add_task(d3.integ(wz_2), name='wz_2')

# main analysis tasks
analysis = solver.evaluator.add_file_handler('analysis_debug_512_256_changeF', sim_dt = 0.1)

# profile
vortm0 = d3.Average(vort, coords['phi'])
analysis.add_task(vortm0, name = 'w0')
drvortm0 = d3.Average(er@d3.grad(vort), coords['phi'])
analysis.add_task(drvortm0, name='drw0')
urm0 = d3.Average(ur, coords['phi'])
analysis.add_task(urm0, name='ur0')
analysis.add_task(urm0 * drvortm0, name='ur0_drw0')

# scalar 
analysis.add_task(d3.integ(vort), name = 'W')
analysis.add_task(drvortm0(r=1), name='drw0_1')
#Fphim0 = d3.Average(ephi@F, coords['phi'])
#analysis.add_task(-Fphim0(r=1), name='F_1')

# coeff space
analysis.add_task(tau_u, layout='c', name='tau_u')
analysis.add_task(u*vort, layout='c', name='u_wz_c') 
analysis.add_task(F, layout='c', name='F_c')
#analysis.add_task(d3.lap(u), layout='c', name='lap_u_c')
analysis.add_task(u, layout='c', name='u_c') 

# snapshots
analysis.add_task(vort, layout='g', name='vorticity')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')

CFL = d3.CFL(solver, initial_dt=tstep, cadence=10, safety=0.15, threshold=0.05, max_dt=5e-4)
CFL.add_velocity(u)

# create checkpoint
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt = 25, max_writes = 1)
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

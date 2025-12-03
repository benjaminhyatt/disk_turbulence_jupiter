"""
Run simulations of disk turbulence in the gamma plane approximation

Usage:
    disk_turb_run.py [options]

Options:    
    --seed=<int>                random seed for stochastic forcing

    --alpha=<float>             large-scale friction [default: 1e-2]
    --gamma=<float>             strength of gamma effect (2 \Omega / a_p^2) [default: 3e1]
    --eps=<float>               energy injection rate (time-averaged) [default: 1e0]
    --nu=<float>                kinematic viscosity [default: 2e-4]
    --kf=<int>                  integer wavenumber (per unit radius) of the forcing [default: 10]

    --Nphi=<int>                azimuthal resolution [default: 512]
    --Nr=<int>                  radial resolution [default: 256]

    --ring=<bool>               turns off forcing as r approaches the boundary [default: False]
    --width=<float>             sets width of the transition region in forcing as r approaches the boundary [default: 0.08] (only used if ring True)

    --restart=<bool>            flag that this run starts from a previous checkpoint (with same parameters) [default: False]
    --restart_evolved=<bool>    indicate in output names that this run starts from a checkpoint from a run with different parameters [default: False]
    --restart_eps_0=<bool>      flag that this run starts from a previous checkpoint, but turning the forcing off [default: False]
    --restart_hyst=<bool>       flag that this run starts from a previous checkpoint, from a different gamma [default: False]
    --hystn=<int>               experiment number [default: 1]
    --restart_dir=<path>        path of checkpoint to restart from [default: None]

    --tau_mod=<bool>            flag False to use default lift operator, True to use suggested modification [default: True]
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.Get_size()

from dedalus.tools.logging import *
logger = logging.getLogger(__name__)

# Parameters
dealias = 3/2
dtype = np.float64

from docopt import docopt
args = docopt(__doc__)

logger.info("args read in")
if rank == 0:
    print(args)

# load in seed -- use below

alpha = float(args['--alpha'])
gamma = float(args['--gamma'])
eps = float(args['--eps'])
amp = np.sqrt(eps)
nu = float(args['--nu'])
k_force = int(2 * int(args['--kf'])) # integer periods along a cart axis
logger.info("k_force, adjusted: %d" %(k_force))

Nphi = int(args['--Nphi'])
Nr = int(args['--Nr'])

ring = eval(args['--ring'])
width = float(args['--width'])

restart = eval(args['--restart'])
restart_evolved = eval(args['--restart_evolved'])
restart_eps_0 = eval(args['--restart_eps_0'])
restart_hyst = eval(args['--restart_hyst'])
hystn= int(args['--hystn'])
if restart or restart_evolved or restart_eps_0 or restart_hyst:
    restart_dir = args['--restart_dir']

tau_mod = eval(args['--tau_mod'])

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) 
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis

#phi, r = dist.local_grids(disk, scales=(dealias, dealias))
#phi_deal, r_deal = dist.local_grids(disk, scales=(1, 1)) #need this size for the force
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

F= d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_func)
#angF= rvec@d3.skew(F)
#Fvort= d3.div(d3.skew(F))

G = dist.Field(bases=disk)
G.preset_scales(dealias)
G['g']= - 0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) #Coriolis parameter
G=d3.Grid(G)

pvort = vort + G

# Problem
problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
if tau_mod:
    if restart_eps_0:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) - alpha*u - G*d3.skew(u)")
    else:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
else:
    if restart_eps_0:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u) - alpha*u - G*d3.skew(u)")
    else:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")

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
stop_time = 8/alpha
timestepper = d3.SBDF2
tstep = 1e-5

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_time
logger.info('solver built')

# Restart from checkpoint
if restart_hyst:
    write, initial_timestep = solver.load_state(restart_dir)
    file_handler_mode = 'overwrite'
    rand = np.random.RandomState(seed=17+solver.iteration)
    output_suffix += '_restart_hyst_{:d}'.format(hystn)
elif restart_evolved:
    #write, initial_timestep = solver.load_state('checkpoints_nu_2em04_gam_0d0ep00_kf_2ep01_Nphi_512_Nr_256_ring_0/checkpoints_nu_2em04_gam_0d0ep00_kf_2ep01_Nphi_512_Nr_256_ring_0_s28.h5')
    write, initial_timestep = solver.load_state(restart_dir)
    file_handler_mode = 'overwrite'
    rand = np.random.RandomState(seed=10001+solver.iteration)
elif restart_eps_0:
    write, initial_timestep = solver.load_state(restart_dir)
    file_handler_mode = 'append'
elif restart:
    #write, initial_timestep = solver.load_state('checkpoints_' + output_suffix +'/checkpoints_' + output_suffix + '_s20.h5')
    write, initial_timestep = solver.load_state(restart_dir)
    file_handler_mode = 'append'
    rand = np.random.RandomState(seed=93+solver.iteration)
else:
    file_handler_mode = 'overwrite'

# Analysis
#analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = 0.1, mode=file_handler_mode)
#analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = 0.02, mode=file_handler_mode)
analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = 0.05, mode=file_handler_mode)
# scalars
analysis.add_task(d3.Average(0.5*u@u), name = 'KE')
analysis.add_task(d3.Average(angm), name = 'Lzu')
analysis.add_task(d3.Average(vort), name = 'W')
analysis.add_task(d3.Average(vort*vort), name = 'EN')
#analysis.add_task(d3.Average(Fvort), name = 'FW')
#analysis.add_task(d3.Average(angF), name = 'LzF')
analysis.add_task(d3.Average(-2 * ((ephi@u)(r=1))**2, coords['phi']), name = 'ENbdry')
analysis.add_task(d3.Average(d3.lap(u)@d3.lap(u)), name = 'PA')
analysis.add_task(d3.Average(-2 * (ephi@u)(r=1) * (er@d3.grad(er@d3.grad((ephi@u))))(r=1), coords['phi']), name='PAbdry1')
analysis.add_task(d3.Average(2 * (ephi@u)(r=1) * (er@d3.grad(ephi@d3.grad((er@u))))(r=1), coords['phi']), name='PAbdry2')

# profiles
vortm0 = d3.Average(vort, coords['phi'])
analysis.add_task(vortm0, name = 'vortm0')
pvortm0 = d3.Average(pvort, coords['phi'])
analysis.add_task(pvortm0, name = 'pvortm0')
drvortm0 = er@d3.grad(vortm0)
analysis.add_task(drvortm0, name = 'drvortm0')
drpvortm0 = er@d3.grad(pvortm0)
analysis.add_task(drpvortm0, name = 'drpvortm0')
#dr2vortm0 = er@d3.grad(drvortm0)
#analysis.add_task(dr2vortm0, name = 'dr2vortm0')
dr2pvortm0 = er@d3.grad(drpvortm0)
analysis.add_task(dr2pvortm0, name = 'dr2pvortm0')

#um0 = d3.Average(u, coords['phi'])
#analysis.add_task(um0, name = 'um0')
#u2m0 = d3.Average(u@u , coords['phi'])
#analysis.add_task(u2m0, name='u2m0')

# snapshots
analysis.add_task(vort, layout='g', name='vort')
analysis.add_task(u, layout='g', name='u')
analysis.add_task(pvort, layout='g', name='pvort')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
flow.add_property(vort*vort, name = 'w2')

CFL = d3.CFL(solver, initial_dt=tstep, cadence=10, safety=0.1, threshold=0.05, max_dt=1e-4) #safety=0.15
CFL.add_velocity(u)

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints_' + output_suffix, sim_dt = 1, max_writes = 1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        tstep = CFL.compute_timestep()
        solver.step(tstep)
        if (solver.iteration-1) % 100 == 0:
            max_u = np.sqrt(flow.max('u2'))
            ke_avg = flow.grid_average('u2')
            en_avg = flow.grid_average('w2')
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e, Kavg=%e, Zavg=%e" %(solver.iteration, solver.sim_time, tstep, max_u, ke_avg, en_avg))
            if max_u > 1e4 or np.isnan(max_u):
                print(max_u)
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

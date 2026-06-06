"""
Run simulations of disk turbulence in the gamma plane approximation

Usage:
    disk_turb_run_inputs.py [options]

Options:    
    --seed=<int>                random seed for stochastic forcing [default: 31415926]
    --flip=<bool>               flip the sign of the vorticity in the generated initial condition [default: False]

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
    --restart_dir=<path>        path of checkpoint to restart from [default: None]
    --restart_hyst=<bool>       flag that this run starts from a previous checkpoint, from a different gamma [default: False]
    --hystn=<int>               experiment number [default: None]

    --tau_mod=<bool>            flag False to use default lift operator, True to use suggested modification [default: True]

    --restart_eps_0=<bool>      flag that this run starts from a previous checkpoint, but turning the forcing off [default: False]
    --restart_evolved=<bool>    indicate in output names that this run starts from a checkpoint from a different run [default: False]

    --redo=<bool>               flag to indicate that this will start from 0 but is aiming to fill in a lost chunk of data in a run we have the later data for [default: False]

    --safety=<float>            CFL safety factor [default: 0.1]
    --timestepper=<string>      Choice of timestepper [default: SBDF2]

    --bc=<string>               Specification of boundary conditions ('sf' or 'ns') [default: sf]
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

from dedalus.tools.config import config

#config['analysis']['FILEHANDLER_PARALLEL_DEFAULT'] = 'gather'
print(config['analysis']['FILEHANDLER_PARALLEL_DEFAULT'])

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

from fractions import Fraction

logger.info("args read in")
if rank == 0:
    print(args)

seed_in = int(args['--seed'])
flip = eval(args['--flip'])

alpha = float(args['--alpha'])
gamma = float(args['--gamma'])
eps = float(args['--eps'])
amp = np.sqrt(eps)
nu = float(Fraction(args['--nu']))
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
if restart_hyst:
    hystn= int(args['--hystn'])

if restart or restart_evolved or restart_eps_0 or restart_hyst:
    restart_dir = args['--restart_dir']

tau_mod = eval(args['--tau_mod'])

safety = float(args['--safety'])
timestepper_str = args['--timestepper']

bc_str = args['--bc']

redo = eval(args['--redo'])

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) 
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_seed_{:d}'.format(seed_in)
output_suffix += '_safety_{:.1e}'.format(safety)
output_suffix += '_timestepper_' + timestepper_str
output_suffix += '_bc_' + bc_str
if redo:
    output_suffix += '_redo_{:d}'.format(redo)
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
# nphi!=0
problem.add_equation("radial(u(r=1)) = 0", condition='nphi!=0')
if bc_str == 'sf':
    problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi!=0')
elif bc_str == 'ns':
    problem.add_equation("azimuthal(u(r=1)) = 0", condition='nphi!=0')
else:
    logger.info('invalid specification of bc - not implemented')
# nphi==0
problem.add_equation("radial(u(r=1)) = 0", condition='nphi==0')
try:
    problem.equations[-1]['valid_modes'][1] = True
except:
    logger.info("Skipping valid modes line on rank %d" %(rank))
if bc_str == 'sf':
    problem.add_equation("azimuthal(radial(stress(r=1))) = 0", condition='nphi==0')
elif bc_str == 'ns':
    problem.add_equation("azimuthal(u(r=1)) = 0", condition='nphi==0')
else:
    logger.info('invalid specification of bc - not implemented')
try:
    problem.equations[-1]['valid_modes'][1] = True
except:
    logger.info("Skipping valid modes line on rank %d" %(rank))
# gauge choice
problem.add_equation("integ(p) = 0")

# timestepping
#stop_time = 8/alpha
stop_time = 1/alpha
if timestepper_str.upper() == 'SBDF2':
    timestepper = d3.SBDF2
elif timestepper_str.upper() == 'RK443':
    timestepper = d3.RK443
elif timestepper_str.upper() == 'RK222':
    timestepper = d3.RK222

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_time
logger.info('solver built')

# restart from checkpoint
if restart or restart_evolved or restart_eps_0 or restart_hyst:
    write, initial_timestep = solver.load_state(restart_dir)
    tstep = initial_timestep
    if restart:
        file_handler_mode = 'append'
    else:
        file_handler_mode = 'overwrite'
    # advance random state forward:
    if timestepper_str == 'SBDF2':
        for old_iter in range(int(2*solver.iteration)):
            phases = rand.uniform(0,2*np.pi,k_len) # 2 new evaluations of forcing_func per iteration
    #rand = np.random.RandomState(seed=seed_in+solver.iteration)

    if flip:
        u['g'] *= -1.
 
    solver.stop_sim_time = solver.sim_time + 30.1 #+ 10.01

else:
    file_handler_mode = 'overwrite'
    tstep = 1e-5

if redo:
    if gamma == 1200:
        stop_time = 98.01 
    elif gamma == 1920:
        stop_time = 106.01 
    elif gamma == 2500:
        stop_time = 109.01
    solver.stop_sim_time = stop_time

# typical Rossby wave time scale
rossby_freq_def = lambda k, m, gam: m * gam * k**(-2)
from scipy.special import jn_zeros
k01 = jn_zeros(1, 2)[0]
rossby_freq_est = rossby_freq_def(k01, 1, gamma)
rossby_period_est = 2*np.pi/rossby_freq_est
# analysis cadence
#period_exp = np.floor(np.log10(np.abs(rossby_period_est)))
#period_round = np.round(rossby_period_est / 10**period_exp, 0) * 10**period_exp
#sim_dt_choice = np.min((0.05, 0.1 * period_round))
freq_exp = np.floor(np.log10(np.abs(rossby_freq_est)))
freq_round = np.round(rossby_freq_est / 10**freq_exp, 0) * 10**freq_exp
sim_dt_choice = np.min((0.05, 1/freq_round)) # 1/freq_round/2))
logger.info("Rossby period estimate: %e, analysis cadence: %e" %(rossby_period_est, sim_dt_choice))

#if not restart:
#    analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = 0.05, mode=file_handler_mode)
#else:
#    analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, custom_schedule=sched, mode=file_handler_mode)
analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt=sim_dt_choice, mode=file_handler_mode)


# scalars
analysis.add_task(d3.Average(0.5*u@u), name = 'KE')
analysis.add_task(d3.Average(angm), name = 'Lzu')
analysis.add_task(d3.Average(vort), name = 'W')
analysis.add_task(d3.Average(vort*vort), name = 'EN')
#analysis.add_task(d3.Average(Fvort), name = 'FW')
#analysis.add_task(d3.Average(angF), name = 'LzF')
#analysis.add_task(d3.Average(-2 * ((ephi@u)(r=1))**2, coords['phi']), name = 'ENbdry')
#analysis.add_task(d3.Average(d3.lap(u)@d3.lap(u)), name = 'PA')
#analysis.add_task(d3.Average(-2 * (ephi@u)(r=1) * (er@d3.grad(er@d3.grad((ephi@u))))(r=1), coords['phi']), name='PAbdry1')
#analysis.add_task(d3.Average(2 * (ephi@u)(r=1) * (er@d3.grad(ephi@d3.grad((er@u))))(r=1), coords['phi']), name='PAbdry2')

#r_scal = dist.Field(bases=disk)
#r2_scal = dist.Field(bases=disk)
#rcos_scal = dist.Field(bases=disk)
#rsin_scal = dist.Field(bases=disk)
#r_scal.change_scales(dealias)
#r2_scal.change_scales(dealias)
#rcos_scal.change_scales(dealias)
#rsin_scal.change_scales(dealias)
#r_scal['g'] = r_deal * pow(phi_deal, 0)
#r2_scal['g'] = pow(r_deal, 2) * pow(phi_deal, 0)
#rcos_scal['g'] = r_deal * np.cos(phi_deal)
#rsin_scal['g'] = r_deal * np.sin(phi_deal)
#om0density = r_scal * vort
#om1cdensity = rcos_scal * vort
#om1sdensity = -rsin_scal * vort
#analysis.add_task(d3.Average(om0density), name = 'om0')
#analysis.add_task(d3.Average(om1cdensity), name = 'om1c')
#analysis.add_task(d3.Average(om1sdensity), name = 'om1s')
#nu0density = r_scal * 2 * nu * d3.lap(vort)
#analysis.add_task(d3.Average(nu0density), name = 'nu0') # I want to see if we get a closed result with just the linear terms in Eq. (1) (i.e., rhs does not contribute in avg)
#nu1cdensity = rcos_scal * 2 * nu * d3.lap(vort)
#analysis.add_task(d3.Average(nu1cdensity), name = 'nu1c')
#nu1sdensity = rsin_scal * 2 * nu * d3.lap(vort)
#analysis.add_task(d3.Average(nu1sdensity), name = 'nu1s')
# if I want to track averages of psi itself, then I will need to solve some LBVPs... I would rather do that in post 

# profiles
um0 = d3.Average(ephi@u, coords['phi'])
analysis.add_task(um0, name = 'um0') # dpsi_0/dr
vortm0 = d3.Average(vort, coords['phi'])
analysis.add_task(vortm0, name = 'vortm0')
pvortm0 = d3.Average(pvort, coords['phi'])
analysis.add_task(pvortm0, name = 'pvortm0')
drvortm0 = er@d3.grad(vortm0) # dlappsi_0/dr
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

max_dt_choice = 1e-4
logger.info('setting max_dt as %e' %(max_dt_choice))
CFL = d3.CFL(solver, initial_dt=tstep, cadence=5, safety=safety, threshold=0.05, max_dt=max_dt_choice)
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

"""
Run simulations of disk turbulence in the gamma plane approximation -- this version enables updates of the gamma parameter at intermediate times

Usage:
    disk_turb_run_gamma.py [options]

Options:    
    
    --gammai=<float>            (initial) strength of gamma effect (2 \Omega / a_p^2) [default: 2.4e2]
    --gammaf=<float>            (final) [default: 3.2e2]
    --dgamma=<float>            step size in gamma to evolve with [default: 4e1]
    --ke_dt=<float>             time interval over which to measure changes in ke [default: 2.5e1]
    --ke_rtol_minmax=<float>  tolerance (relative to ke avg over ke_dt) for |ke_max - ke_min| over ke_dt [default: 2e-1]
    --ke_rtol_lin=<float>     tolerance (relative to ke avg over ke_dt) for linear fit's dke [default: 1e-1]

    --seed=<int>                random seed for stochastic forcing [default: 31415926]

    --alpha=<float>             large-scale friction [default: 1e-2]
    --eps=<float>               energy injection rate (time-averaged) [default: 1e0]
    --nu=<float>                kinematic viscosity [default: 2e-4]
    --kf=<int>                  integer wavenumber (per unit radius) of the forcing [default: 10]

    --Nphi=<int>                azimuthal resolution [default: 512]
    --Nr=<int>                  radial resolution [default: 256]

    --tau_mod=<bool>            flag False to use default lift operator, True to use suggested modification [default: True]

    --ring=<bool>               turns off forcing as r approaches the boundary [default: False]
    --width=<float>             sets width of the transition region in forcing as r approaches the boundary [default: 0.08] (only used if ring True)

    --restart=<bool>            flag that this run starts from a previous checkpoint (with same parameters) [default: False]
    --restart_dir=<path>        path of checkpoint to restart from [default: None]
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.Get_size()

from dedalus.tools.logging import *
logger = logging.getLogger(__name__)

from scipy.optimize import curve_fit

# Parameters
dealias = 3/2
dtype = np.float64

from docopt import docopt
args = docopt(__doc__)

logger.info("args read in")
if rank == 0:
    print(args)

gammai = float(args['--gammai'])
gammaf = float(args['--gammaf'])
dgamma = float(args['--dgamma'])
ke_dt = float(args['--ke_dt'])
ke_rt_mm = float(args['--ke_rtol_minmax'])
ke_rt_lin = float(args['--ke_rtol_lin'])

seed_in = int(args['--seed'])

alpha = float(args['--alpha'])
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
if restart:
    restart_dir = args['--restart_dir']

tau_mod = eval(args['--tau_mod'])

output_suffix = 'nu_{:.0e}'.format(nu)
output_suffix += '_gami_{:.1e}'.format(gammai)
output_suffix += '_gamf_{:.1e}'.format(gammaf)
output_suffix += '_dgam_{:.1e}'.format(dgamma)
output_suffix += '_kf_{:.1e}'.format(k_force)
output_suffix += '_Nphi_{:}'.format(Nphi)
output_suffix += '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_seed_{:d}'.format(seed_in)
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

def setup_G(gamma):
    g1 = dist.Field(bases=disk)
    g1.preset_scales(dealias)
    g1['g'] = -0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) 
    g1=d3.Grid(g1)
    return g1

def linear_curve(xdata, a1, a2):
    return a1*xdata + a2

def build_and_run_solver(experiment_vars, init_vars):
    
    ## unpack input args
    gamma, tf, rtol_mm, rtol_lin = experiment_vars
    use_state, state_dir, file_handler_mode, analysis_dir, checkpoints_dir = init_vars

    ## build Coriolis vars
    G = build_G(gamma)
    pvort = vort + G

    ## build problem 
    problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
    problem.add_equation("div(u) + tau_p = 0")
    if tau_mod:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
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

    ## build solver
    timestepper = d3.SBDF2
    logger.info('building solver')
    solver = problem.build_solver(timestepper)
    logger.info('solver built')

    ## initial state (either load a state or fields will be zero)
    if use_state:
        write, tstep = solver.load_state(state_dir)
        rand = np.random.RandomState(seed=seed_in+solver.iteration)
    else:
        tstep = 1e-5
        rand = np.random.RandomState(seed=seed_in)

    ngamma = np.ceil((gammaf - gammai) / dgamma)
    solver.stop_sim_time = solver.sim_time + ngamma * (2/alpha) # should typically never take longer than this

    ## analysis 'analysis_' + output_suffix
    analysis = solver.evaluator.add_file_handler(analysis_dir, sim_dt = 0.05, mode=file_handler_mode)
    # scalars
    analysis.add_task(d3.Average(0.5*u@u), name = 'KE')
    analysis.add_task(d3.Average(angm), name = 'Lzu')
    analysis.add_task(d3.Average(vort), name = 'W')
    analysis.add_task(d3.Average(vort*vort), name = 'EN')
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
    dr2pvortm0 = er@d3.grad(drpvortm0)
    analysis.add_task(dr2pvortm0, name = 'dr2pvortm0')
    # snapshots
    analysis.add_task(vort, layout='g', name='vort')
    analysis.add_task(u, layout='g', name='u')
    analysis.add_task(pvort, layout='g', name='pvort')

    ## checkpoints 'checkpoints_' + output_suffix
    checkpoints = solver.evaluator.add_file_handler(checkpoints_dir, sim_dt = 1, max_writes = 1, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)

    ## flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(u@u, name='u2')
    flow.add_property(vort*vort, name = 'w2')
    
    ## cfl
    CFL = d3.CFL(solver, initial_dt=tstep, cadence=10, safety=0.1, threshold=0.05, max_dt=1e-4)
    CFL.add_velocity(u)

    ## evolve
    dt_measure = np.max((ke_dt / 1e3, max_dt))
    t_last_measure = 0.
    ke_vals = []
    ke_ts = []

    t_begin_check = solver.sim_time + ke_dt
    dt_check = ke_dt / 2
    t_last_check = 0.

    logger.info("Start time: %e, time to begin checking: %e, gamma: %e" %(solver.sim_time, t_begin_check, gamma)) 
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            
            tstep = CFL.compute_timestep()
            solver.step(tstep)
            
            # experiment measurement
            if solver.sim_time >= t_last_measure + dt_measure:
                ke_avg = flow.grid_average('u2')
                ke_vals.append(float(ke_avg))
                ke_ts.append(float(solver.sim_time))
                t_last_measure = solver.sim_time

            # experiment check
            if solver.sim_time >= t_begin_check:
                if solver.sim_time >= t_last_check + dt_check:
                    ke_min = np.min(ke_vals)
                    ke_max = np.max(ke_vals)
                    ke_fit, ~ = curve_fit(linear_curve, ke_ts, ke_vals)
                    
                    ke_tavg = linear_curve(t_last_check + dt_check/2, ke_fit[0], ke_fit[1])
                    
                    dke_mm = np.max((np.abs(ke_max - ke_tavg), np.abs(ke_min - ke_tavg)))
                    dke_mm_rel = dke_mm / ke_tavg

                    dke_lin = ke_fit[0] * dt_check
                    dke_lin_rel = np.abs(dke_lin) / ke_tavg

                    t_last_check = solver.sim_time
                    if dke_mm_rel <= ke_rtol_mm:
                        logger.info("exit1 passed: dke_mm_rel = %e, ke_rtol_mm = %e" %(dke_mm_rel, ke_rtol_mm))
                        exit1 = True
                    if dke_mm_rel <= ke_rtol_mm:
                        logger.info("exit1 failed: dke_mm_rel = %e, ke_rtol_mm = %e" %(dke_mm_rel, ke_rtol_mm))
                        exit1 = False 
                    if dke_lin_rel <= ke_rtol_lin:
                        logger.info("exit2 passed: dke_lin_rel = %e, ke_rtol_lin = %e" %(dke_lin_rel, ke_rtol_lin))
                        exit2 = True
                    else:
                        logger.info("exit2 failed: dke_lin_rel = %e, ke_rtol_lin = %e" %(dke_lin_rel, ke_rtol_lin))
                        exit2 = False

                    if exit1 and exit2:
                        
                        # def returns here ? 

                        logger.info("breaking loop on solver.iteration = %i" %(solver.iteration))
                        solver.proceed = False

                    else: 
                        logger.info("one or more checks did not pass, proceeding")

                        # trim ke_vals and ke_ts lists here

            # typical outputs/checks
            if ((solver.iteration-1) % 100 == 0) or (not solver.proceed):
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

    return # tbd


# todo: main driver of the experiment here -- either a while or for loop that steps through changes in gamma, defines input args, etc.




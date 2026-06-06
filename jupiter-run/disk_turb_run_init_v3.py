"""
Run simulations of disk turbulence in the gamma plane approximation.

v3 changes relative to v2:
  * 'nonlin' analysis task now saves the *vector* u@grad(u) instead of the
    scalar u@grad(vort). This is the object needed by the post-processing
    T(k,t) = -<u, u@grad(u)> projection.
  * Three conservation diagnostics:
      L  = integral of (r * u_phi) over the disk (total angular momentum;
           should stay at zero by construction of the forcing & IC, and
           cleanly isolates the m=0 H=-1 Dini initial term -- the J_1
           Robin modes contribute exactly zero to L).
      Px = integral of u_x over the disk
      Py = integral of u_y over the disk
    Px, Py track linear momentum.  With the m_J = |m-1| convention they
    are now PROXIES for the m=1 initial-term content of u_phi (they mix
    the initial-term content with regular Bessel-mode u_r content), but
    still useful as a physical conservation diagnostic.
  * Four direct initial-term diagnostics, exact in-IVP versions of the
    post-processing D_init_{cos,sin} projections:
      D_init_1_cos, D_init_1_sin  for the m=1 sector (basis r^0 = 1)
      D_init_2_cos, D_init_2_sin  for the m=2 sector (basis r^1 = r)
    The m=0 initial term (basis r^1) is already covered by L.  These four
    plus L give rigorous, one-number-per-snapshot monitoring of every
    initial-term mode the H=-1 Dini expansion would otherwise drop.
  * lap(u) snapshot saved at grid layout, for post-processing use of the
    full vector Laplacian when computing the viscous dissipation budget
    (since the scalar 2*nu*lambda^2*E approximation misses the
    -u/r^2 metric + (2/r^2) d_phi cross-coupling terms of (lap u)_r,phi).

Usage:
    disk_turb_run_init.py [options]

Options:
    --seed=<int>                random seed for stochastic forcing [default: 31415926]
    --flip=<bool>               flip the sign of the vorticity in the generated initial condition [default: False]

    --alpha=<float>             large-scale friction [default: 1e-2]
    --gamma=<float>             strength of gamma effect (2 \Omega / a_p^2) [default: 3e1]
    --eps=<float>               energy injection rate (time-averaged) [default: 1e0]
    --nu=<float>                kinematic viscosity [default: 2e-4]
    --kf=<int>                  integer wavenumber (per unit radius) of the forcing [default: 10]

    --kinit=<float>               wavenumber (per unit radius) of the initial condition [default: 10]
                                  (for the transient decay parameter study, set equal to --kf)

    --Nphi=<int>                azimuthal resolution [default: 512]
    --Nr=<int>                  radial resolution [default: 256]

    --ring=<bool>               turns off forcing as r approaches the boundary [default: False]
    --width=<float>             sets width of the transition region in forcing as r approaches the boundary [default: 0.04] (only used if ring True)

    --restart=<bool>            flag that this run starts from a previous checkpoint (with same parameters) [default: False]
    --restart_dir=<path>        path of checkpoint to restart from [default: None]

    --tau_mod=<bool>            flag False to use default lift operator, True to use suggested modification [default: False]
    --ring_init=<bool>          turns off energy in initial condition as r approaches the boundary [default: True]

    --restart_evolved=<bool>    indicate in output names that this run starts from a checkpoint from a run with different parameters [default: False]

    --safety=<float>            CFL safety factor [default: 0.1]
    --timestepper=<string>      Choice of timestepper [default: SBDF2]

    --ncc_cutoff=<float>        value of ncc_cutoff [default: 1e-6]
    --implicit=<bool>           flag True to treat Coriolis term implicitly [default: False]
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

from fractions import Fraction

seed_in = int(args['--seed'])
flip = eval(args['--flip'])

alpha = float(args['--alpha'])
gamma = float(args['--gamma'])
eps = float(args['--eps'])
amp = np.sqrt(eps)
nu = float(Fraction(args['--nu']))
k_force = int(2 * int(args['--kf'])) # integer periods along a cart axis
logger.info("k_force, adjusted: %d" %(k_force))

k_init = 2 * float(args['--kinit'])
logger.info("k_init, adjusted: %e" %(k_init))

Nphi = int(args['--Nphi'])
Nr = int(args['--Nr'])

ring = eval(args['--ring'])
width = float(args['--width'])

restart = eval(args['--restart'])
restart_evolved = eval(args['--restart_evolved'])
if restart or restart_evolved:
    restart_dir = args['--restart_dir']

tau_mod = eval(args['--tau_mod'])
ring_init = eval(args['--ring_init'])

safety = float(args['--safety'])
timestepper_str = args['--timestepper']

ncc_cutoff = float(args['--ncc_cutoff'])
implicit = eval(args['--implicit'])

output_suffix = 'nu_{:.0e}'.format(nu)
output_suffix += '_gam_{:.1e}'.format(gamma)
output_suffix += '_kf_{:.1e}'.format(k_force)
output_suffix += '_ki_{:.1e}'.format(k_init)
output_suffix += '_Nphi_{:}'.format(Nphi)
output_suffix += '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_ring_init_{:d}'.format(ring_init)
output_suffix += '_seed_{:d}'.format(seed_in)
output_suffix += '_safety_{:.1e}'.format(safety)
output_suffix += '_timestepper_' + timestepper_str
output_suffix += '_ncc_cutoff{:.1e}'.format(ncc_cutoff)
output_suffix += '_implicit_{:d}'.format(implicit)
if flip:
    output_suffix += '_flip_{:d}'.format(flip)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

# Steady-state area-averaged KE estimate in the nu->0 limit.
# Energy balance: eps (total injection) = 2*alpha*KE_total = 2*alpha*pi*KE_avg
# => KE_avg = d3.Average(0.5*u@u) = eps / (2 * alpha * pi)
norm_goal_ss = eps / (2.0 * alpha * np.pi)

# urms estimate derived from nu->0 steady-state KE: urms = sqrt(2 * KE_avg)
urms_est_ss = np.sqrt(2.0 * norm_goal_ss)

# Bases

# Disk
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk, scales=(1, 1))
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
phi_deal_g, r_deal_g = disk.global_grids(dist, scales=(dealias, dealias))

# Cartesian
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

def make_transform(k_in):
    mask = (k >= k_in) & (k < k_in+1)
    x_disk = r_deal*np.cos(phi_deal)
    y_disk = r_deal*np.sin(phi_deal)

    transform = 2*np.exp(1j*kx[mask][None,:]*x_disk.flatten()[:,None]
                        + 1j*ky[mask][None,:]*y_disk.flatten()[:,None])
    phi_2d = (phi_deal + 0*r_deal).ravel()
    transform_vector_rotated = np.stack( (1j*ky[mask][None,:]*transform, -1j*kx[mask][None,:]*transform ) ) / k_in**2
    transform_vector = np.stack( (-transform_vector_rotated[0]*np.sin(phi_2d[:,None]) + transform_vector_rotated[1]*np.cos(phi_2d[:,None]),
                               transform_vector_rotated[0]*np.cos(phi_2d[:,None]) + transform_vector_rotated[1]*np.sin(phi_2d[:,None])))

    k_len = len(k[mask])
    phi_len = len(phi_deal.ravel())
    r_len = len(r_deal.ravel())
    transform_vector = transform_vector.reshape(2, phi_len, r_len, k_len)
    return k_len, transform_vector

k_len_forcing, transform_vector_forcing = make_transform(k_force)
k_len_init, transform_vector_init = make_transform(k_init)

rand = np.random.RandomState(seed=seed_in) #10001
phases_forcing = rand.uniform(0,2*np.pi,k_len_forcing)
forcing = np.real(transform_vector_forcing @ np.exp(1j*phases_forcing))

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

# Cartesian unit-vector fields in polar components for linear-momentum diagnostics.
# In polar: e_x = cos(phi) e_r - sin(phi) e_phi
#          e_y = sin(phi) e_r + cos(phi) e_phi
# Dedalus convention: VectorField['g'][0] is the phi-component, [1] is the r-component.
ex_field = dist.VectorField(coords, bases=disk)
ey_field = dist.VectorField(coords, bases=disk)
ex_field['g'][0] = -np.sin(phi + 0*r)
ex_field['g'][1] =  np.cos(phi + 0*r)
ey_field['g'][0] =  np.cos(phi + 0*r)
ey_field['g'][1] =  np.sin(phi + 0*r)

# Scalar fields used in the initial-term D_init_{1,2}_{cos,sin} diagnostics.
# These are the "test functions" against which we project u_phi to extract
# the H=-1 Dini initial-term coefficients at m=1 (basis r^0 = 1) and
# m=2 (basis r^1 = r).
cos_phi   = dist.Field(bases=disk)
sin_phi   = dist.Field(bases=disk)
cos_2phi  = dist.Field(bases=disk)
sin_2phi  = dist.Field(bases=disk)
r_scalar  = dist.Field(bases=disk)
cos_phi['g']  = np.cos(phi + 0*r)
sin_phi['g']  = np.sin(phi + 0*r)
cos_2phi['g'] = np.cos(2 * (phi + 0*r))
sin_2phi['g'] = np.sin(2 * (phi + 0*r))
r_scalar['g'] = r + 0*phi

vort = -d3.div(d3.skew(u))
angm = -rvec@d3.skew(u)

v1=dist.VectorField(coords, bases=disk)
v1.preset_scales(dealias)

# Define GeneralFunction for forcing
def forcing_func():
    phases = rand.uniform(0,2*np.pi,k_len_forcing)
    f1=np.real(transform_vector_forcing @ np.exp(1j*phases))
    if ring:
        f1 *= 0.5*(1-np.tanh( (r_deal-0.75)/width ))
    v1['g'] = f1
    angf1=d3.integ(rvec@d3.skew(v1)).evaluate()

    phases2 = rand.uniform(0,2*np.pi,k_len_forcing)
    f2=np.real(transform_vector_forcing @ np.exp(1j*phases2))
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

def forcing_budget_func():
    return forcing_func() * np.sqrt(tstep)
F_budget = d3.GeneralFunction(dist, u.domain, u.tensorsig, dtype, 'g', forcing_budget_func)


if not implicit:
    G = dist.Field(bases=disk)
    G.preset_scales(dealias)
    G['g']= - 0.5*gamma*pow(r_deal,2) * pow(phi_deal,0) #Coriolis parameter
    G=d3.Grid(G)
else:
    G = dist.Field(bases=radial_basis)
    G.preset_scales(dealias)
    G['g']= - 0.5*gamma*pow(r_deal,2)
pvort = vort + G

# Problem
problem = d3.IVP([p, u, tau_u, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
if not implicit:
    if tau_mod:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) + sig*lift_2(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
    else:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u) + amp*F - alpha*u - G*d3.skew(u)")
else:
    if tau_mod:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + G*d3.skew(u) + lift(tau_u) + sig*lift_2(tau_u) + alpha*u = - u@grad(u) + amp*F")
    else:
        problem.add_equation("dt(u) - nu*lap(u) + grad(p) + G*d3.skew(u) + lift(tau_u) + alpha*u = - u@grad(u) + amp*F")
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

# Timestepping
# Run for ~3 friction times to ensure the transient decays to steady state.
stop_time = 3.0 / alpha

if timestepper_str.upper() == 'SBDF2':
    timestepper = d3.SBDF2
elif timestepper_str.upper() == 'RK443':
    timestepper = d3.RK443
elif timestepper_str.upper() == 'RK222':
    timestepper = d3.RK222

# Solver
logger.info('building solver')
solver = problem.build_solver(timestepper, ncc_cutoff=ncc_cutoff)
solver.stop_sim_time = stop_time
logger.info('solver built')

# CFL
umax_est = np.sqrt(3 * eps/alpha)
urms_est = urms_est_ss
logger.info('urms_est (nu->0 ss estimate) %e' %(urms_est))
if gamma > 0:
    lgam_est = (urms_est / gamma) ** (1/3)
else:
    lgam_est = 0.5
drphi_lgam = lgam_est * np.mean(np.gradient(phi_deal_g[:, 0]))
dr_lgam = np.gradient(r_deal_g[0, :])[np.where(r_deal_g[0, :] <= lgam_est)[0][-1]]

denom_init = (umax_est/np.sqrt(2))*(1/drphi_lgam) + (umax_est/np.sqrt(2))*(1/dr_lgam)
denom_long = (urms_est/np.sqrt(2))*(1/drphi_lgam) + (urms_est/np.sqrt(2))*(1/dr_lgam)
max_dt_choice_init = np.min((2 * safety / denom_init, safety / denom_long))
max_dt_choice_long = safety / denom_long
t_switch = 1e0

logger.info('using max_dt of %e up until t=%e, then max_dt will be set to %e' %(max_dt_choice_init, t_switch, max_dt_choice_long))
tstep = max_dt_choice_init

CFL = d3.CFL(solver, initial_dt=tstep, cadence=5, safety=safety, threshold=0.05, max_dt=max_dt_choice_init)
CFL.add_velocity(u)
CFL_switch = d3.CFL(solver, initial_dt=max_dt_choice_init, cadence=5, safety=safety, threshold=0.05, max_dt=max_dt_choice_long)
CFL_switch.add_velocity(u)

# Handle restart from checkpoint
if restart or restart_evolved:
    write, initial_timestep = solver.load_state(restart_dir)
    if restart:
        file_handler_mode = 'append'
    else:
        file_handler_mode = 'overwrite'
    rand = np.random.RandomState(seed=seed_in+solver.iteration)
    #solver.stop_sim_time = solver.sim_time + 1.01
else:
    file_handler_mode = 'overwrite'

# Analysis cadence
if gamma > 0:
    # typical Rossby wave time scale
    rossby_freq_def = lambda k, m, gam: m * gam * k**(-2)
    rossby_freq_est = rossby_freq_def(2*np.pi, 1, gamma)
    rossby_period_est = 2*np.pi/rossby_freq_est
    period_exp = np.floor(np.log10(np.abs(rossby_period_est)))
    period_round = np.round(rossby_period_est / 10**period_exp, 0) * 10**period_exp
    sim_dt_choice = np.min((0.02, 0.1 * period_round))
    logger.info("Rossby period estimate: %e, analysis cadence: %e" %(rossby_period_est, sim_dt_choice))
else:
    sim_dt_choice = 0.02
    logger.info("gamma = 0, analysis cadence: %e" %(sim_dt_choice))

logger.info("using sim_dt = %e" %(sim_dt_choice))
analysis = solver.evaluator.add_file_handler('analysis_' + output_suffix, sim_dt = sim_dt_choice, mode=file_handler_mode)

# scalars
analysis.add_task(d3.Average(0.5*u@u), name = 'KE')
analysis.add_task(d3.Average(vort), name = 'W')
analysis.add_task(d3.Average(vort*vort), name = 'EN')

# v3 conservation diagnostics:
# L  is the *total* angular momentum, integral of (r*u_phi) over the disk.
#    Should remain numerically zero by construction of the forcing (zero
#    angular impulse) and initial condition (zero net angular momentum).
# Px, Py are the *total* linear momentum components, integrals of u_x, u_y.
#    Not explicitly constrained, but should fluctuate around zero with
#    amplitude small compared to sqrt(KE)*disk_area scales.  These quantify
#    how much energy could be hiding in the H=-1 Dini initial-term modes
#    (m=0 rigid rotation -> L; m=1 uniform translation -> Px, Py).
analysis.add_task(d3.integ(angm),       name = 'L')
analysis.add_task(d3.integ(u@ex_field), name = 'Px')
analysis.add_task(d3.integ(u@ey_field), name = 'Py')

# Direct initial-term diagnostics for the H=-1 Dini expansion of u_phi.
# Formulas (derived in the v3 design discussion):
#   D_init_m^{cos/sin} = (2(m_J+1) / pi) * integ(u_phi * trig(m*phi) * r^{m_J})
# For m=1: m_J = |m-1| = 0,  basis r^0 = 1,  factor 2/pi
# For m=2: m_J = |m-1| = 1,  basis r^1 = r,  factor 4/pi
# The m=0 initial term (basis r^1) is already given by  D_init_0 = (2/pi) * L.
analysis.add_task((2.0/np.pi) * d3.integ((u@ephi) * cos_phi),               name = 'D_init_1_cos')
analysis.add_task((2.0/np.pi) * d3.integ((u@ephi) * sin_phi),               name = 'D_init_1_sin')
analysis.add_task((4.0/np.pi) * d3.integ((u@ephi) * cos_2phi * r_scalar),   name = 'D_init_2_cos')
analysis.add_task((4.0/np.pi) * d3.integ((u@ephi) * sin_2phi * r_scalar),   name = 'D_init_2_sin')

# profiles
um0 = d3.Average(u@ephi, coords['phi'])
analysis.add_task(um0, name = 'um0')
vortm0 = d3.Average(vort, coords['phi'])
analysis.add_task(vortm0, name = 'vortm0')
pvortm0 = d3.Average(pvort, coords['phi'])
#analysis.add_task(pvortm0, name = 'pvortm0')
drvortm0 = er@d3.grad(vortm0)
#analysis.add_task(drvortm0, name = 'drvortm0')
drpvortm0 = er@d3.grad(pvortm0)
#analysis.add_task(drpvortm0, name = 'drpvortm0')
dr2vortm0 = er@d3.grad(drvortm0)
#analysis.add_task(dr2vortm0, name = 'dr2vortm0')
dr2pvortm0 = er@d3.grad(drpvortm0)
analysis.add_task(dr2pvortm0, name = 'dr2pvortm0')

# snapshots
analysis.add_task(vort, layout='g', name='vort')
analysis.add_task(u, layout='g', name='u')
# v3 change: 'nonlin' is now the *vector* u@grad(u), not the scalar u@grad(vort).
# This is the field that gets dotted with u in post to form T(k,t) = -<u, u@grad(u)>.
analysis.add_task(u@d3.grad(u), layout='g', name='nonlin')
analysis.add_task(u@d3.grad(vort), layout='g', name='nonlin_vort')
# v3 addition: full vector Laplacian of u, saved for post-processing use of
# the exact viscous dissipation 2*nu * integ(|grad u|^2) per k-bin, which
# the scalar-Laplacian approximation 2*nu*lambda^2*E misses by the
# -u/r^2 metric and (2/r^2) d_phi cross-coupling terms.
analysis.add_task(d3.lap(u), layout='g', name='lap_u')
analysis.add_task(amp*F_budget, layout='g', name='forcing')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
flow.add_property(vort*vort, name = 'w2')

# Checkpoints
checkpoints = solver.evaluator.add_file_handler('checkpoints_' + output_suffix, sim_dt = 0.25, max_writes = 1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# Non-trivial initial condition
v_init = dist.VectorField(coords, bases=disk)
v_init.preset_scales(dealias)
u_init = dist.VectorField(coords, bases=disk)
u_init.preset_scales(dealias)

if not restart:
    norm_goal = norm_goal_ss  # eps / (2 * alpha * pi), nu->0 steady-state estimate
    nseeds = 10
    randinit_1 = np.random.RandomState(seed=int(seed_in*2))
    randinit_2 = np.random.RandomState(seed=int(seed_in*3))
    for n in range(int(nseeds/2)):
        phasesinit_1 = randinit_1.uniform(0,2*np.pi,k_len_init)
        init_1 = np.real(transform_vector_init @ np.exp(1j*phasesinit_1))
        if ring_init:
            init_1 *= 0.5*(1-np.tanh( (r_deal-0.75)/width ))
        v_init['g'] = init_1
        anginit_1=d3.integ(rvec@d3.skew(v_init)).evaluate()
        if rank == 0:
            data = [anginit_1['g'][0][0]]*size
        else:
            data = None
        anginit_1_int= MPI.COMM_WORLD.scatter(data, root=0)

        phasesinit_2 = randinit_2.uniform(0,2*np.pi,k_len_init)
        init_2 = np.real(transform_vector_init @ np.exp(1j*phasesinit_2))
        if ring_init:
            init_2 *= 0.5*(1-np.tanh( (r_deal-0.75)/width ))
        v_init['g'] = init_2
        anginit_2=d3.integ(rvec@d3.skew(v_init)).evaluate()
        if rank == 0:
            data = [anginit_2['g'][0][0]]*size
        else:
            data = None
        anginit_2_int= MPI.COMM_WORLD.scatter(data, root=0)

        u_init['g'] += init_1/anginit_1_int - init_2/anginit_2_int

    norm = d3.Average(0.5*u_init@u_init).evaluate()
    if rank == 0:
        data = [norm['g'][0][0]]*size
    else:
        data = None
    norm_int = MPI.COMM_WORLD.scatter(data, root=0)
    u.preset_scales(dealias)
    u['g'] = u_init['g'] * np.sqrt(norm_goal/norm_int)
    if flip:
        u['g'] *= -1.
elif flip:
    u['g'] *= -1.

initial_energy = d3.Average(0.5*u@u).evaluate()
if rank == 0:
    data = [initial_energy['g'][0][0]]*size
else:
    data = None
initial_energy_int= MPI.COMM_WORLD.scatter(data, root=0)
logger.info("Initial energy: %e" %(initial_energy_int))

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        if solver.sim_time < t_switch:
            tstep = CFL.compute_timestep()
        else:
            tstep = CFL_switch.compute_timestep()
        solver.step(tstep)
        if (solver.iteration-1) % 1000 == 0:
            max_u = np.sqrt(flow.max('u2'))
            ke_avg = flow.grid_average('u2')
            en_avg = flow.grid_average('w2')
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e, Kavg=%e, Zavg=%e" %(solver.iteration, solver.sim_time, tstep, max_u, ke_avg, en_avg))
            if max_u > 1e4 or np.isnan(max_u):
                print("max_u break", max_u)
                break
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

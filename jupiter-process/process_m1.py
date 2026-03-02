"""
Usage:
    process_m1.py <file>... [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_m1]
"""
import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
from dedalus.extras import plot_tools
import scipy as sp
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
### read arguments passed to script ###
print("args read in")
print(args)

file_str = args['<file>'][0]

output_prefix = args['--output']

### read analysis file ###
f = h5py.File(file_str)

### string parsing to identify parameters ###
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

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
gamma_read = str_to_float(gamma_str)
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]

# load in analysis file
f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

# dedalus setup 
dealias = 3/2
Nphi_deal = int(np.round(dealias * Nphi))
Nr_deal = int(np.round(dealias * Nr))
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
vort = dist.Field(name = 'vort', bases = disk)
# polar to cartesian conversion
plot_func = lambda phi, r, data: (r*np.cos(phi), r*np.sin(phi), data)

Phi, R = plot_tools.quad_mesh(phi_deal[:, 0], r_deal[0, :])

# Rossby profile
def rossby(X, A, B, om): 
    # fixed params
    m = 1 
    g = gamma
    # independent vars
    ph = X[:Nphi_deal]
    ra = X[Nphi_deal:]
    ph = np.array(ph).reshape(Nphi_deal, 1)
    ra = np.array(ra).reshape(1, Nr_deal)
    # wavenumber
    k = np.sqrt(g * m / om)
    z = sp.special.jvp(m, k * ra, n=2) * (A * np.cos(m*ph) - B * np.sin(m*ph))
    return z.ravel()
# arguments to pass to curve_fit
lower_bds = [-np.inf, -np.inf, 0.]
upper_bds = [np.inf, np.inf, np.inf]
bds = (lower_bds, upper_bds)
indep_vars = np.concatenate((phi_deal.ravel(), r_deal.ravel()))
pars0 = (1., 1., gamma/(4*np.pi**2))

# range of writes
ws = np.arange(len(t))
nw = len(ws)
tw = t[ws] # w = 0 corresponds to t = 0

# append results
phis = np.zeros(nw)
Z_origs = np.zeros((nw, Nphi_deal, Nr_deal))
Z_m1s = np.zeros((nw, Nphi_deal, Nr_deal))
Z_fits = np.zeros((nw, Nphi_deal, Nr_deal))
fit_pars = np.zeros((nw, 3))
### Loop over writes
prog_cad = 32
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        logger.info("Outer writes loop: i = %d out of %d" %(i, nw))

    # load in vorticity data from corresponding write
    vort.load_from_hdf5(f, w)

    # grid data as-is
    vort.change_scales(dealias)
    vortg = np.copy(vort['g'])

    # m = 1 projection 
    vortm1 = dist.Field(name = 'vortm1', bases = disk)
    vortc_m1 = np.copy(vort['c'][4:6, :]) 
    vortm1['c'][4:6, :] += vortc_m1
    vortm1.change_scales(dealias)
    vortm1g = np.copy(vortm1['g'])

    # call curve_fit
    pars, covs = sp.optimize.curve_fit(rossby, indep_vars, vortm1g.ravel(), p0=pars0, bounds=bds)

   
    Z_origs[i, :, :] = np.copy(vortg)
    Z_m1s[i, :, :] = np.copy(vortm1g)
    Z_fits[i, :, :] = np.copy(rossby(indep_vars, pars[0], pars[1], pars[2]).reshape(Nphi_deal, Nr_deal))
    fit_pars[i, :] = pars

    # naive choice of phi to plot slice through
    fit_max_phi_idx, fit_max_r_idx = np.unravel_index(np.argmax(Z_fits[i, :, :]), Z_fits[i, :, :].shape)
    phis[i] = fit_max_phi_idx    

processed = {}
processed['ws'] = ws
processed['ts'] = tw
processed['Phi'] = Phi
processed['R'] = R
processed['r'] = r_deal
processed['phi_seq'] = phis
processed['Z_orig'] = Z_origs
processed['Z_m1'] = Z_m1s
processed['Z_fit'] = Z_fits
processed['fit_pars'] = fit_pars

print('saving processed results as: ' + output_prefix + '_' + output_suffix + '.npy')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

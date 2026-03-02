"""
Make plots to inspect the m=1 component of the vorticity, and compare to to predictions for Rossby wave profiles. 

Usage:
    plot_m1.py <file>... [options]

Options:    
    --profiles=<bool>       true to generate profile plots [default: True]
    --pcolormesh=<bool>     true to generate pcolormesh plots [default: True]
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import h5py
from dedalus.extras import plot_tools
from docopt import docopt
args = docopt(__doc__)
import scipy as sp

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

print("args read in")
print(args)

file_str = args['<file>'][0]

# may need to comment out the last split if loading in a virtually merged file (something to fix)
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] 

do_profiles = eval(args['--profiles'])
do_pcolormesh = eval(args['--pcolormesh'])

# determine parameter values
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
gamma_read = str_to_float(gamma_str)
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]

output_dir = 'm1_plots_gam_{:.1e}'.format(gamma) 
output_dir = output_dir.replace('+','p').replace('.','d')

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

# fit projection to rossby
lower_bds = [-np.inf, -np.inf, 0.]
upper_bds = [np.inf, np.inf, np.inf]
bds = (lower_bds, upper_bds)

indep_vars = np.concatenate((phi_deal.ravel(), r_deal.ravel()))

pars0 = (1., 1., gamma/(4*np.pi**2))

tlook0 = 400
tidx = np.where(t >= tlook0)[0][0]
nlook = 21

for n in range(nlook):
    # load in vorticity data from corresponding write
    vort.load_from_hdf5(f, tidx)

    # grid data as-is
    vort.change_scales(dealias)
    vortg = np.copy(vort['g'])

    # m = 1 projection 
    vortm1 = dist.Field(name = 'vortm1', bases = disk)
    vortc_m1 = np.copy(vort['c'][4:6, :])
    vortm1['c'][4:6, :] += vortc_m1
    vortm1.change_scales(dealias)
    vortm1g = np.copy(vortm1['g'])

    pars, covs = sp.optimize.curve_fit(rossby, indep_vars, vortm1g.ravel(), p0=pars0, bounds=bds) 
    #pars, covs = sp.optimize.curve_fit(rossby, indep_vars, vortm1g.ravel(), bounds=bds, x_scale = pars0)
    print(tidx, pars)

    # plot comparisons
    Phi, R = plot_tools.quad_mesh(phi_deal[:, 0], r_deal[0, :])
    X = (R * np.cos(Phi)).T
    Y = (R * np.sin(Phi)).T
    Z_orig = vortg
    Z_m1 = vortm1g
    Z_fit = rossby(indep_vars, pars[0], pars[1], pars[2]).reshape(Nphi_deal, Nr_deal)

    if do_pcolormesh:

        fig, ax = plt.subplots(figsize=(7, 6))
        lim = max(abs(Z_orig.min()), abs(Z_orig.max()))
        mesh = ax.pcolormesh(X, Y, Z_orig, shading='auto', cmap='RdBu_r', vmin = -lim, vmax = lim)
        fig.colorbar(mesh, ax=ax)
        plt.savefig(output_dir + '/orig_' + str(tidx) + '_' + output_suffix + '.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        lim = max(abs(Z_m1.min()), abs(Z_m1.max()))
        mesh = ax.pcolormesh(X, Y, Z_m1, shading='auto', cmap='RdBu_r', vmin = -lim, vmax = lim)
        fig.colorbar(mesh, ax=ax)
        plt.savefig(output_dir + '/m1_' + str(tidx) + '_' + output_suffix + '.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        lim = max(abs(Z_fit.min()), abs(Z_fit.max()))
        mesh = ax.pcolormesh(X, Y, Z_fit, shading='auto', cmap='RdBu_r', vmin = -lim, vmax = lim)
        fig.colorbar(mesh, ax=ax)
        plt.savefig(output_dir + '/fit_' + str(tidx) + '_' + output_suffix +'.png')
        plt.close(fig)

    if do_profiles:

        fit_max_phi_idx, fit_max_r_idx = np.unravel_index(np.argmax(Z_fit), Z_fit.shape) 

        plt.figure()
        plt.plot(r_deal[0, :], Z_m1[fit_max_phi_idx, :], color = 'blue', label = r'$m = 1$ data')
        plt.plot(r_deal[0, :], Z_fit[fit_max_phi_idx, :], color = 'orange', label = 'Fit')
        plt.xlabel(r'$r$')
        plt.ylabel('Vorticity along a fixed $\phi$')
        plt.savefig(output_dir + '/prof_' + str(tidx) + '_' + output_suffix + '.png')
        plt.close()

    tidx += 1

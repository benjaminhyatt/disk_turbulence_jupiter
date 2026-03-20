"""
Process data and make plots related to the m=1 components of the vorticity.
(This may be preferable to avoid the intermediate task of saving a large processing file for especially long runs.) 

Usage:
    process_and_plot_m1_panels.py <files>... [--output=<dir> --tracking=<bool> --t_start=<float>]

Options:
    --output=<dir>      output directory [default: ./frames]
    --tracking=<bool>   whether to plot processed tracking locations [default: False]
    --t_start <float>   sim time to begin processing [default: 0.]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import dedalus.public as d3
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

# Rossby profile
#def rossby(x, A, B, sh, om, fixed_args): 
def rossby(x, A, B, om, fixed_args):
    # fixed params
    m = 1 
    #t, g, Nphi_deal, Nr_deal = fixed_args    
    g, Nphi_deal, Nr_deal = fixed_args
    
    # independent vars
    ph = x[:Nphi_deal]
    ra = x[Nphi_deal:]
    ph = np.array(ph).reshape(Nphi_deal, 1)
    ra = np.array(ra).reshape(1, Nr_deal)
    # wavenumber
    k = np.sqrt(g * m / om) 
    #z = sp.special.jvp(m, k * ra, n=2) * (A * np.cos(m*ph) - B * np.sin(m*ph))
    #z = sp.special.jv(m, k * ra) * (A * np.cos(m*ph - om*t - sh) - B * np.sin(m*ph - om*t - sh))
    z = sp.special.jv(m, k * ra) * (A * np.cos(m*ph) - B * np.sin(m*ph))
    return z.ravel()

#def process_m1(vort_field, vortm1_field, phis, rs, time, args):
def process_m1(vort_field, vortm1_field, phis, rs, args):    
    # unpack args
    dealias, gamma, Nphi_deal, Nr_deal = args

    vort_field.change_scales(dealias)
    vortg = np.copy(vort_field['g'])
    
    vortm1_field['c'] *= 0.
    vortc_m1 = np.copy(vort_field['c'][4:6, :]) 
    vortm1_field['c'][4:6, :] += vortc_m1
    vortm1_field.change_scales(dealias)
    vortm1g = np.copy(vortm1_field['g'])

    lower_bds = [-np.inf, -np.inf, 0.] #[-np.inf, -np.inf, 0., 0.]
    upper_bds = [np.inf, np.inf, np.inf]#[np.inf, np.inf, 2*np.pi, np.inf]
    bds = (lower_bds, upper_bds)
    indep_vars = np.concatenate((phis.ravel(), rs.ravel()))
    pars0 = (1., 1., gamma/(4*np.pi**2)) #(1., 1., 0., gamma/(4*np.pi**2))

    #fixed_args = [time, gamma, Nphi_deal, Nr_deal]
    fixed_args = [gamma, Nphi_deal, Nr_deal]
    #fit_func = lambda x, a, b, c, d: rossby(x, a, b, c, d, fixed_args)
    fit_func = lambda x, a, b, c: rossby(x, a, b, c, fixed_args)

    pars, covs = sp.optimize.curve_fit(fit_func, indep_vars, vortm1g.ravel(), p0=pars0, bounds=bds)
    
    #vortfitg = np.copy(rossby(indep_vars, pars[0], pars[1], pars[2], pars[3], fixed_args).reshape(Nphi_deal, Nr_deal))
    vortfitg = np.copy(rossby(indep_vars, pars[0], pars[1], pars[2], fixed_args).reshape(Nphi_deal, Nr_deal))

    fit_max_phi_idx, fit_max_r_idx = np.unravel_index(np.argmax(vortfitg), vortfitg.shape)

    return vortg, vortm1g, vortfitg, fit_max_phi_idx, pars

def main(filename, start, count, output, options):
    """Save plot of specified tasks for given range of analysis writes."""
    
    # unpack options
    plottracking = options

    # determine some parameters from string parsing
    output_suffix = filename.split('analysis_')[1].split('.')[0].split('/')[0]
    Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
    Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
    gamma_str = output_suffix.split('gam_')[1].split('_')[0]
    gamma_read = str_to_float(gamma_str)
    gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
    gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]

    # load file
    f = h5py.File(filename, mode='r')
    times = np.array(f['scales/sim_time'])

    # objects to store fit results
    processed_pars = []
    phi_choices = []

    # dedalus setup for processing
    dealias = 3/2 
    Nphi_deal = int(np.round(dealias * Nphi))
    Nr_deal = int(np.round(dealias * Nr))
    dtype = np.float64
    coords = d3.PolarCoordinates('phi', 'r')
    dist = d3.Distributor(coords, dtype = dtype, comm = MPI.COMM_SELF)
    disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
    edge = disk.edge
    radial_basis = disk.radial_basis
    phi, r = dist.local_grids(disk)
    phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
    vort = dist.Field(name = 'vort', bases = disk)
    vortm1 = dist.Field(name = 'vortm1', bases = disk)

    # arguments to pass to process_m1 func
    args = [dealias, gamma, Nphi_deal, Nr_deal]

    # plotting setup
    dpi = 300
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    Phi, R = plot_tools.quad_mesh(phi_deal[:, 0], r_deal[0, :])
    X = (R * np.cos(Phi)).T
    Y = (R * np.sin(Phi)).T

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.15, 0.25, 0.15, 0.1)
    h_plot, w_plot = (1., 1.)
    h_pad = 0.25 * h_plot
    w_pad = 0.35 * w_plot
    h_cbar = 0.05 * h_plot
    w_cbar = w_plot
    h_total = t_mar + h_plot + h_cbar + b_mar
    w_total = l_mar + w_plot + w_pad + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    # axs and caxs setup #
    left1 = (l_mar) / w_total
    bottom1 = 1 - (t_mar + h_plot) / h_total
    width1 = w_plot / w_total
    height1 = h_plot / h_total

    left2 = (l_mar + w_plot + w_pad) / w_total
    bottom2 = 1 - (t_mar + h_plot) / h_total
    width2 = w_plot / w_total
    height2 = h_plot / h_total

    leftc1 = (l_mar) / w_total
    bottomc1 = 1 - (t_mar + h_plot + h_cbar) / h_total
    widthc1 = w_plot / w_total
    heightc1 = h_cbar / h_total

    leftc2 = (l_mar + w_plot + w_pad) / w_total
    bottomc2 = 1 - (t_mar + h_plot + h_cbar) / h_total
    widthc2 = w_plot / w_total
    heightc2 = h_cbar / h_total

    # Plot writes
    progress_cad = np.ceil(count/20)
    for index in range(start, start+count): 
        if index % progress_cad == 0 and rank == 0:
            logger.info("index = %d, start = %d, start+count=%d" %(index, start, start+count))
        # process current index
        vort.load_from_hdf5(f, index)
        #time = times[index]
        #Z_orig, Z_m1, Z_fit, phi_choice, pars = process_m1(vort, vortm1, phi_deal, r_deal, time, args)
        Z_orig, Z_m1, Z_fit, phi_choice, pars = process_m1(vort, vortm1, phi_deal, r_deal, args)
        processed_pars.append(pars)

        phi_choice_val = phi_deal[phi_choice, 0]
        phi_choices.append(phi_choice_val)

        if plottracking:
            idx_track = index - ws[0]
            x_loc = r_locs[idx_track] * np.cos(phi_locs[idx_track])
            y_loc = r_locs[idx_track] * np.sin(phi_locs[idx_track])

        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        lim1 = max(abs(Z_orig.min()), abs(Z_orig.max()))
        mesh1 = ax1.pcolormesh(X, Y, Z_orig, shading='auto', cmap='RdBu_r', vmin = -lim1, vmax = lim1)
        if plottracking and index in ws:
            ax1.scatter(x_loc, y_loc, color = 'black', lw = 0.5)
        ax1.xaxis.set_visible(False)        
        ax1.yaxis.set_visible(False)
        ax1.set_frame_on(False)
        ax1.set_title('original data')
        cax1 = fig.add_axes([leftc1, bottomc1, widthc1, heightc1])
        cbar1 = fig.colorbar(mesh1, cax=cax1, orientation='horizontal')

        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        lim2 = max(abs(Z_m1.min()), abs(Z_m1.max()))
        mesh2 = ax2.pcolormesh(X, Y, Z_m1, shading='auto', cmap='RdBu_r', vmin = -lim2, vmax = lim2)
        if plottracking and index in ws:
            ax2.scatter(x_loc, y_loc, color = 'black', lw = 0.5)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.set_frame_on(False)
        ax2.set_title(r'$m = 1$ projection')
        cax2 = fig.add_axes([leftc2, bottomc2, widthc2, heightc2])
        cbar2 = fig.colorbar(mesh2, cax=cax2, orientation='horizontal')

        # Add time title
        title = title_func(f['scales/sim_time'][index])
        title_height = 1 - 0.125 * t_mar
        fig.suptitle(title, x=0.44, y=title_height, ha='left')
        # Save figure
        savename = savename_func(index)
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi)
        fig.clear()

    plt.close(fig)

    # output fit results
    out_name = 'processed_m1_v2_' + str(rank)
    processed = {}
    processed['ws'] = np.arange(start, start+count)
    processed['pars'] = processed_pars
    processed['phi_choices'] = phi_choices
    print('rank ' + str(rank) + ' saving output as processed_m1_' + output_suffix + '_' + str(rank) + '_v2.npy')
    np.save(out_name + '.npy', processed)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    from dedalus.tools.logging import *
    logger = logging.getLogger(__name__) 

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()

    plottracking = eval(args['--tracking'])
    t_start = float(args['--t_start'])
    
    if t_start > 0.:
        import post_tstart
    else:
        from dedalus.tools import post

    # string parsing if we want to load in additional files from processing
    if plottracking:
        file_str = args['<files>'][0]
        output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]

        processed_tracking = np.load('../jupiter-process/processed_tracking_' + output_suffix + '.npy', allow_pickle = True)[()]
        r_locs = processed_tracking['r_locs']
        phi_locs = processed_tracking['phi_locs']
        ws = processed_tracking['ws']

    options_in = plottracking
    if t_start > 0.:
        post_tstart.visit_writes(args['<files>'], main, t_start, output=output_path, options=options_in)
    else:
        post.visit_writes(args['<files>'], main, output=output_path, options=options_in)

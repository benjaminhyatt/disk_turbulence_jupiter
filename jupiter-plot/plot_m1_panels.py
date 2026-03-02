"""
Make plots related to the m=1 components of the vorticity.

Usage:
    plot_m1_panels.py <files>... [--output=<dir> --tracking=<bool>]

Options:
    --output=<dir>      output directory [default: ./frames]
    --tracking=<bool>   whether to plot processed tracking locations [default: False]
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

def main(filename, start, count, output, options):
    """Save plot of specified tasks for given range of analysis writes."""

    plottracking = options

    dpi = 300
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.1, 0.1, 0.1, 0.1)
    h_plot, w_plot = (1., 1.)
    h_pad = 0.05 * h_plot
    w_pad = 0.05 * w_plot
    h_cbar = 0.05 * h_plot
    w_cbar = w_plot
    h_total = t_mar + h_plot + h_cbar + h_pad + h_plot + h_cbar + b_mar
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

    left3 = (l_mar) / w_total
    bottom3 = 1 - (t_mar + h_plot + h_cbar + h_pad + h_plot) / h_total
    width3 = w_plot / w_total
    height3 = h_plot / h_total

    left4 = (l_mar + w_plot + w_pad) / w_total
    bottom4 = 1 - (t_mar + h_plot + h_cbar + h_pad + h_plot) / h_total
    width4 = w_plot / w_total
    height4 = h_plot / h_total

    leftc1 = (l_mar) / w_total
    bottomc1 = 1 - (t_mar + h_plot + h_cbar)
    widthc1 = w_plot / w_total
    heightc1 = h_cbar / h_total

    leftc2 = (l_mar + w_plot + w_pad) / w_total
    bottomc2 = 1 - (t_mar + h_plot + h_cbar)
    widthc2 = w_plot / w_total
    heightc2 = h_cbar / h_total

    leftc3 = (l_mar) / w_total
    bottomc3 = 1 - (t_mar + h_plot + h_cbar + h_pad + h_pad + h_plot + h_cbar)
    widthc3 = w_plot / w_total
    heightc3 = h_cbar / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]

    Phi = f['Phi']
    R = f['R']
    X = (R * np.cos(Phi)).T
    Y = (R * np.sin(Phi)).T

    Z_orig = f['Z_orig']
    Z_m1 = f['Z_m1']
    Z_fit = f['Z_fit']
    r = f['r']
    phi_seq = f['phi_seq']

    progress_cad = np.ceil(count/20)

    for index in range(start, start+count): 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        if plottracking:
            idx_track = index - ws[0]
            x_loc = r_locs[idx_track] * np.cos(phi_locs[idx_track])
            y_loc = r_locs[idx_track] * np.sin(phi_locs[idx_track])

        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        lim1 = max(abs(Z_orig[index].min()), abs(Z_orig[index].max()))
        mesh1 = ax.pcolormesh(X, Y, Z_orig[index], shading='auto', cmap='RdBu_r', vmin = -lim1, vmax = lim1)
        if plottracking and index in ws:
            ax1.scatter(x_loc, y_loc, color = 'black', lw = 0.5)
        cax1 = fig.add_axes([leftc1, bottomc1, widthc1, heightc1])
        cbar1 = fig.colorbar(mesh1, ax=cax1)

        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        lim2 = max(abs(Z_m1[index].min()), abs(Z_m1[index].max()))
        mesh2 = ax.pcolormesh(X, Y, Z_m1[index], shading='auto', cmap='RdBu_r', vmin = -lim2, vmax = lim2)
        if plottracking and index in ws:
            ax2.scatter(x_loc, y_loc, color = 'black', lw = 0.5)
        cax2 = fig.add_axes([leftc2, bottomc2, widthc2, heightc2])
        cbar2 = fig.colorbar(mesh2, ax=cax2)

        ax3 = fig.add_axes([left3, bottom3, width3, height3])
        lim3 = max(abs(Z_fit[index].min()), abs(Z_fit[index].max()))
        mesh3 = ax.pcolormesh(X, Y, Z_fit[index], shading='auto', cmap='RdBu_r', vmin = -lim3, vmax = lim3)
        if plottracking and index in ws:
            ax3.scatter(x_loc, y_loc, color = 'black', lw = 0.5)
        cax3 = fig.add_axes([leftc3, bottomc3, widthc3, heightc3])
        cbar3 = fig.colorbar(mesh3, ax=cax3)

        ax4 = fig.add_axes([left4, bottom4, width4, height4])
        if plottracking and index in ws:     
            ax4.plot(r[0, :], Z_m1[index, phi_locs[idx_track], :], color = 'blue', label = r'$m = 1$ data')
            ax4.plot(r[0, :], Z_fit[index, phi_locs[idx_track], :], color = 'orange', label = r'fit')
            ax4.axvline(r_locs[idx_track], linestyle = 'dashed', color = 'black')
        else:
            ax4.plot(r[0, :], Z_m1[index, phi_seq[index], :], color = 'blue', label = r'$m = 1$ data')
            ax4.plot(r[0, :], Z_fit[index, phi_seq[index], :], color = 'orange', label = r'fit')
            
        ax4.set_xlabel(r'$r$')
        ax4.set_ylabel(r'$\omega(r), \phi =$0' + '{:.3e}'.format(phi_seq[index]))
        ax4.legend(loc = 'lower left')

        # Add time title
        title = title_func(f['ts'][index])
        title_height = 1 - 0.125 * t_mar
        fig.suptitle(title, x=0.44, y=title_height, ha='left')
        # Save figure
        savename = savename_func(index)
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi)
        fig.clear()

    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    import post_npy    

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

    # string parsing if we want to load in additional files from processing
    if plottracking:
        file_str = args['<files>'][0]
        output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 

        processed_tracking = np.load('../jupiter-process/processed_tracking_' + output_suffix + '.npy', allow_pickle = True)[()]
        r_locs = processed_tracking['r_locs']
        phi_locs = processed_tracking['phi_locs']
        ws = processed_tracking['ws']

    options_in = plottracking

    post_npy.visit_writes(args['<files>'], main, output=output_path, options=options_in)

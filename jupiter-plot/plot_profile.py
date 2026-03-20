"""
Make plots of radial profiles

Usage:
    plot_profiles.py <files>... [--output=<dir>]

Options:
    --output=<dir>      output directory [default: ./frames]
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

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    dpi = 300
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.1, 0.2, 0.35, 0.1)
    golden_mean = (np.sqrt(5) - 1.) / 2.
    h_plot, w_plot = (1., 1. / golden_mean)
    h_pad = 0.25 * h_plot
    h_total = t_mar + h_plot + b_mar
    w_total = l_mar + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    # axs setup #
    left1 = (l_mar) / w_total
    bottom1 = 1 - (t_mar + h_plot) / h_total
    width1 = w_plot / w_total
    height1 = h_plot / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]
    r = f['r']
    um0 = f['um0']
    um0_tavg = f['um0_tavg']

    normalize = np.sqrt(1/0.033) ### in the future, do this systematically
    kf = 20 * 2 * np.pi# same comment
    eps = 1 # same comment
    output_suffix = filename.split('processed_profiles_')[1]

    progress_cad = np.ceil(count/20)

    #for index in range(start, start+count): 
    #    if index % progress_cad == 0 and rank == 0:
    #        frac = (index)/count
    #        percent = '{:.3}'.format(100*frac)
    #        print("Rank {:}:".format(rank), percent, "% complete")

        # profiles of vorticity and potential vorticity
    #    ax1 = fig.add_axes([left1, bottom1, width1, height1])
    #    ax1.axhline(0., color = 'gray')
    #    ax1.plot(r, um0_tavg / normalize, color = 'black', linewidth = 2.25, label = r'$u$ time average')
    #    ax1.plot(r, um0[index] / normalize, color = 'blue', linewidth = 1.5, label = r'$u$')
    #    ax1.set_xlabel(r'$r$')
    #    ax1.set_ylabel(r'$u(r)/\sqrt{\epsilon/\alpha}$')
    #    ax1.set_xscale('log')
    #    ax1.legend(loc = 'lower left')

        # Add time title
    #    title = title_func(f['tw'][index])
    #    title_height = 1 - 0.125 * t_mar
    #    fig.suptitle(title, x=0.44, y=title_height, ha='left')
        # Save figure
    #    savename = savename_func(index)
    #    savepath = output.joinpath(savename)
    #    fig.savefig(str(savepath), dpi=dpi)
    #    fig.clear()

    #plt.close(fig)

    # Make a plot of just time-averaged profiles 
    if rank == 0:
        fig = plt.figure(figsize = (scale * w_total, scale * h_total))
        # profiles of vorticity and potential vorticity
        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        ax1.plot(r, um0_tavg / normalize, color = 'black', linewidth = 2.25)
        ax1.set_xlabel(r'$r$')
        ax1.set_ylabel(r'$u(r)/\sqrt{\epsilon/\alpha}$ (time-averaged)')
        ax1.set_xscale('log')
        #ax1.legend(loc = 'lower left')
        # Save figure
        savepath = 'tavg_' + output_suffix + '.png'
        print('saving as', savepath)
        fig.savefig(str(savepath), dpi=dpi)
        fig.clear()

        plt.close(fig)

        #fig = plt.figure(figsize = (scale * w_total, scale * h_total))
        # profiles of vorticity and potential vorticity
        #ax1 = fig.add_axes([left1, bottom1, width1, height1])
        #ax1.plot(r, um0_tavg / r, color = 'blue', linewidth = 1.75, label = r'$(u(r)/r)$')
        #ax1.axhline(eps**(1/3)*kf**(2/3), color = 'purple', linestyle = 'dashed', label = r'$\tau_{\rm ed}^{-1} = \epsilon^{1/3}k_f^{2/3}$')
        #ax1.axvline(2 * np.pi / kf, color = 'grey', linestyle = 'dashed', label = r'$r = 2\pi/k_f$')
        #ax1.set_xlabel(r'$r$')
        #ax1.set_xscale('log')
        #ax1.legend(loc = 'upper right')
        ## Save figure
        #savepath = 'tavg_v2_' + output_suffix + '.png'
        #print('saving as', savepath)
        #fig.savefig(str(savepath), dpi=dpi)
        #fig.clear()


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

    post_npy.visit_writes(args['<files>'], main, output=output_path)

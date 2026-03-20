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
    h_total = t_mar + h_plot + h_pad + h_plot + h_pad + h_plot + b_mar
    w_total = l_mar + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    # axs setup #
    left1 = (l_mar) / w_total
    bottom1 = 1 - (t_mar + h_plot) / h_total
    width1 = w_plot / w_total
    height1 = h_plot / h_total

    left2 = (l_mar) / w_total
    bottom2 = 1 - (t_mar + h_plot + h_pad + h_plot) / h_total
    width2 = w_plot / w_total
    height2 = h_plot / h_total

    left3 = (l_mar) / w_total
    bottom3 = 1 - (t_mar + h_plot + h_pad + h_plot + h_pad + h_plot) / h_total
    width3 = w_plot / w_total
    height3 = h_plot / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]
    r = f['r']
    vortm0 = f['vortm0']
    vortm0_tavg = f['vortm0_tavg']
    pvortm0 = f['pvortm0']
    pvortm0_tavg = f['pvortm0_tavg']
    drvortm0 = f['drvortm0']
    drvortm0_tavg = f['drvortm0_tavg']
    drvortm0 = f['drvortm0']
    drvortm0_tavg = f['drvortm0_tavg']
    drpvortm0 = f['drpvortm0']
    drpvortm0_tavg = f['drpvortm0_tavg']
    dr2pvortm0 = f['dr2pvortm0']
    dr2pvortm0_tavg = f['dr2pvortm0_tavg']

    progress_cad = np.ceil(count/20)

    for index in range(start, start+count): 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        # profiles of vorticity and potential vorticity
        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        ax1.axhline(0., color = 'gray')
        ax1.plot(r, vortm0_tavg, color = 'black', linewidth = 2.25, label = r'$\omega$ time average')
        ax1.plot(r, vortm0[index], color = 'blue', linewidth = 1.5, label = r'$\omega$')
        ax1.plot(r, pvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$q$ time average')
        ax1.plot(r, pvortm0[index], color = 'orange', linewidth = 1.5, label = r'$q$')
        ax1.set_xlabel(r'$r$')
        ax1.legend(loc = 'lower left')

        # profiles of their first radial derivatives
        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        ax2.axhline(0., color = 'gray')
        ax2.plot(r, drvortm0_tavg, color = 'black', linewidth = 2.25, label = r'$\partial_r \omega$ time average')
        ax2.plot(r, drvortm0[index], color = 'blue', linewidth = 1.5, label = r'$\partial_r \omega$')
        ax2.plot(r, drpvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$\partial_r q$ time average')
        ax2.plot(r, drpvortm0[index], color = 'orange', linewidth = 1.5, label = r'$\partial_r q$')
        ax2.set_xlabel(r'$r$')
        ax2.legend(loc = 'lower left')

        # profiles of the second radial derivative of pv
        ax3 = fig.add_axes([left3, bottom3, width3, height3])
        ax3.plot(r, dr2pvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$\partial_r^2 q$ time average')
        ax3.plot(r, dr2pvortm0[index], color = 'orange', linewidth = 1.5, label = r'$\partial_r^2 q$')
        ax3.set_xlabel(r'$r$')
        ax3.set_yscale('symlog', linthresh = 1e4)
        ax3.legend(loc = 'lower left')

        # Add time title
        title = title_func(f['tw'][index])
        title_height = 1 - 0.125 * t_mar
        fig.suptitle(title, x=0.44, y=title_height, ha='left')
        # Save figure
        savename = savename_func(index)
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi)
        fig.clear()

    plt.close(fig)

    # Make a plot of just time-averaged profiles 
    if rank == 0:
        fig = plt.figure(figsize = (scale * w_total, scale * h_total))
        # profiles of vorticity and potential vorticity
        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        ax1.plot(r, vortm0_tavg, color = 'black', linewidth = 2.25, label = r'$\omega$')
        ax1.plot(r, pvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$q = \omega - \frac{1}{2}\gamma r^2$')
        ax1.set_xlabel(r'$r$')
        ax1.set_ylabel('time-averaged profile')
        ax1.legend(loc = 'lower left')

        # profiles of their first radial derivatives
        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        ax2.plot(r, drvortm0_tavg, color = 'black', linewidth = 2.25, label = r'$\partial_r \omega$')
        ax2.plot(r, drpvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$\partial_r q$')
        ax2.set_xlabel(r'$r$')
        ax2.set_ylabel('time-averaged profile')
        ax2.legend(loc = 'lower left')

        # profiles of the second radial derivative of pv
        ax3 = fig.add_axes([left3, bottom3, width3, height3])
        ax3.plot(r, dr2pvortm0_tavg, color = 'red', linewidth = 2.25, label = r'$\partial_r^2 q$')
        ax3.set_xlabel(r'$r$')
        ax3.set_ylabel('time-averaged profile')
        ax3.set_yscale('symlog', linthresh = 1e4)
        ax3.legend(loc = 'lower left')

        # Save figure
        savepath = 'profiles_tavg_' + args['--output'] + '.png'
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

    post_npy.visit_writes(args['<files>'], main, output=output_path)

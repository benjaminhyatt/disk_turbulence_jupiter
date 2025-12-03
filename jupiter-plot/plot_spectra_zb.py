"""
Make frames of azimuthally-integrated spectra

Usage:
    plot_spectra_zb.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames] 
"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import transforms
from dedalus.extras import plot_tools

from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(filename, start, count, output):
    
    plot_tavg = True
    #task_tavg = 'keBn_tavg'

    trunc_data = True
    trunc_scale = 128 #sig/2pi approx Nr/2

    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.15, 0.2, 0.275, 0.05)
    #golden_mean = (np.sqrt(5) - 1.) / 2.
    #h_plot, w_plot = (1., 1. / golden_mean)
    h_plot, w_plot = (1., 1.)
    h_pad = b_mar
    h_total = t_mar + h_plot + b_mar
    w_total = l_mar + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    ##### construct axs #####
    left1 = (l_mar) / w_total
    bottom1 = 1 - (t_mar + h_plot) / h_total
    width1 = w_plot / w_total
    height1 = h_plot / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]
    nframes = len(f['ws'])

    tasks = ['keBn_zonal', 'keBn_nz', 'keBn']
    #tasks = ['keBn_zonal', 'keBn_nz']

    labels = {}
    labels['keBn'] = r'$K_{\rm tot}$'
    labels['enBn'] = r'$Z_{\rm tot}$'
    labels['keBn_zonal'] = r'$K_{m = 0}$'
    labels['enBn_zonal'] = r'$Z_{m = 0}$'
    labels['keBn_nz'] = r'$K_{m \ne 0}$'
    labels['enBn_nz'] = r'$Z_{m \ne 0}$'

    colors = ['#7570b3', '#1b9e77', '#d95f02']

    ymin = 10**(np.floor(np.log10(np.min(f[tasks[-1]][1:,:]))))
    ymax = 10**(np.ceil(np.log10(np.max(f[tasks[-1]][1:,:]))))

    progress_cad = np.ceil(count/20)

    for index in range(start, start+count):
 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        xdata = f['centers'] / (2 * np.pi)
        
        for p, task in enumerate(tasks):
            ydata = f[task][index, :]
            ydatap = np.copy(ydata[np.abs(ydata) > 1e-14])
            xdatap = np.copy(xdata[np.abs(ydata) > 1e-14])
            if trunc_data:
                ydatap = np.copy(ydatap[xdatap <= trunc_scale])
                xdatap = np.copy(xdatap[xdatap <= trunc_scale])
            ax1.plot(xdatap, ydatap, "-o", markersize = 2.5, linewidth = 1.5, label = labels[task], color = colors[p])
            if plot_tavg:
                ydata = f[task + '_tavg'][:]
                ydata_tavg = np.copy(ydata[np.abs(ydata) > 1e-14])
                xdata_tavg = np.copy(xdata[np.abs(ydata) > 1e-14])
                if trunc_data:
                    ydata_tavg = np.copy(ydata_tavg[xdata_tavg <= trunc_scale])
                    xdata_tavg = np.copy(xdata_tavg[xdata_tavg <= trunc_scale])
                ax1.plot(xdata_tavg, ydata_tavg, linestyle = "dashed", dashes = (5, 6), color = colors[p])
        #if plot_tavg:
        #    ydata = f[task_tavg][:]
        #    ax1.plot(xdata, ydata, markersize = 2.5, linestyle = "dashed", dashes = (5, 6), color = "black")

        ax1.set_xlabel(r'$\sigma / 2\pi$')
        #ax1.set_ylabel(labels[task])
        ax1.set_ylim(ymin, ymax)
        
        ax1.legend(loc = "lower left", fontsize = 8)

        # Add time title
        title = title_func(f['ts'][index])
        title_height = 1 - 0.25 * t_mar
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
    import post_npy
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        output_path = './frames'

    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post_npy.visit_writes(args['<files>'], main, output=output_path)

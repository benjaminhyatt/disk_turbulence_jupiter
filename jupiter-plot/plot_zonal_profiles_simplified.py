"""
Make frames of profiles

Usage:
    plot_profiles.py <files>... [--output=<dir>]

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
    
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.3, 0.1)
    golden_mean = (np.sqrt(5) - 1.) / 2.
    h_plot, w_plot = (1., 1. / golden_mean)
    h_pad = b_mar
    h_total = t_mar + h_plot + h_pad + h_plot + h_pad + h_plot + b_mar
    w_total = l_mar + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    ##### construct axs #####
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
    nframes = f['nout']
    tasks = f['tasks']
    subtasks = f['subtasks']
    labels = f['labels']
    ymin_u = f['ymin_u']
    ymax_u = f['ymax_u']
    ymin_vort = f['ymin_vort']
    ymax_vort = f['ymax_vort']

    #print(ymin_u, ymax_u, ymin_vort, ymax_vort)
    yabsmax_u = np.max(np.abs([ymin_u, ymax_u]))
    yabsmax_vort = np.max(np.abs([ymin_vort, ymax_vort]))
    

    progress_cad = np.ceil(count/20)

    colors = ['#d95f02', '#7570b3', '#1b9e77']

    for index in range(start, start+count):
 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        
        task_idx = 0
        task = tasks[task_idx]
        xdata = f[index][task]['phi']

        m = 0
        subtask = subtasks[task][m]
        ydata = f[index][task]['data_' + subtask].ravel()
        ax1.plot(xdata, ydata, linewidth = 1.5, linestyle = "solid", label = labels[task][m])
        ax1.legend(loc = "lower left", fontsize = 8)

        ax2 = fig.add_axes([left2, bottom2, width2, height2])

        m = 2 
        subtask = subtasks[task][m]
        ydata = f[index][task]['data_' + subtask].ravel()
        ax2.plot(xdata, ydata, linewidth = 1.5, linestyle = "solid", label = labels[task][m])
        ax2.legend(loc = "lower left", fontsize = 8)

        ax3 = fig.add_axes([left3, bottom3, width3, height3])

        m = 4
        subtask = subtasks[task][m]
        ydata = f[index][task]['data_' + subtask].ravel()
        ax3.plot(xdata, ydata, linewidth = 1.5, linestyle = "solid", label = labels[task][m])
        ax3.legend(loc = "lower left", fontsize = 8)


        # Add time title
        title = title_func(f[index][task]['t'])
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

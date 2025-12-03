"""
Plot disk outputs.

Usage:
    plot_disk.py <files>... [--output=<dir> --pvort=<bool> --rms=<bool> --max=<bool> --ea=<bool>]

Options:
    --output=<dir>  Output directory [default: ./frames]
    --pvort=<bool> whether to plot pvort [default: False]
    --rms=<bool> whether to plot Lgamma from u_rms [default: False]
    --max=<bool> whether to plot Lgamma from u_max [default: False]
    --ea=<bool> whether to plot Lgamma from u_ea (i.e., sqrt(eps/alpha)) [default: False]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

def main(filename, start, count, output, options):
    """Save plot of specified tasks for given range of analysis writes."""
    plotpvort, plotgam, plotrms, plotmax, plotea = options
    
    if plotpvort:
        tasks = ['vort', 'pvort']
        #tasks = ['vorticity', 'pv']
    else:
        tasks = ['vort']
        #tasks = ['vorticity']

    cmap = plt.cm.RdBu_r
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    dpi = 200
    func = lambda phi, r, data: (r*np.cos(phi), r*np.sin(phi), data)

    # Layout
    if plotpvort:
        nrows, ncols = 2, 1
    else:
        nrows, ncols = 1, 1
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0, 0, 0, 0)
    margin = plot_tools.Frame(0.1, 0.1, 0.1, 0.1)
    scale = 3

    # Create figure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plotting loop
    progress_cad = np.ceil(count/100)
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                
                # dsets
                dset = file['tasks'][task]
                dset_max = np.max(dset[index])
                dset_phis = np.array(dset.dims[1][0])
                dset_rs = np.array(dset.dims[2][0])
                
                # main plot
                paxes, caxes = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)

                # overlay circle
                if plotgam:
                    if plotrms:                    
                        circ_x = Lgams_u_rms[index] * np.cos(dset_phis)
                        circ_y = Lgams_u_rms[index] * np.sin(dset_phis)
                        paxes.plot(circ_x, circ_y, color = "purple")
                        circ_x = Lgam_u_rms_tavg * np.cos(dset_phis)
                        circ_y = Lgam_u_rms_tavg * np.sin(dset_phis)
                        paxes.plot(circ_x, circ_y, color = "purple", linestyle = "dotted", label = r'$r_{\rm rms} = (\langle u_{\rm rms}\rangle / \gamma)^{1/3}$')
                    if plotmax:                    
                        circ_x = Lgams_u_max[index] * np.cos(dset_phis)
                        circ_y = Lgams_u_max[index] * np.sin(dset_phis)
                        paxes.plot(circ_x, circ_y, color = "orange")
                        circ_x = Lgam_u_max_tavg * np.cos(dset_phis)
                        circ_y = Lgam_u_max_tavg * np.sin(dset_phis)
                        paxes.plot(circ_x, circ_y, color = "orange", linestyle = "dotted", label = r'$r_{\rm max} = (\langle u_{\rm max}\rangle / \gamma)^{1/3}$')
                    if plotea:                    
                        circ_x = Lgam_u_ea_tavg * np.cos(dset_phis)
                        circ_y = Lgam_u_ea_tavg * np.sin(dset_phis)
                        paxes.plot(circ_x, circ_y, color = "black", linestyle = "dotted", label = r'$r_{\epsilon, \alpha} = (\sqrt{\epsilon/\alpha} / \gamma)^{1/3}$')
                
                    paxes.legend(loc = 'lower left', fontsize = 6)
                
                paxes.axis('off')
                caxes.cla()
                caxes.axis('off')
            
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            if plotrms or plotmax or plotea:
                title += '\n'
            if plotrms: 
                title += r', $r_{\rm rms}$' + ' = {:.3f}'.format(Lgams_u_rms[index])
            if plotmax: 
                title += r', $r_{\rm max}$' + ' = {:.3f}'.format(Lgams_u_max[index])
            if plotea: 
                title += r', $r_{\epsilon, \alpha}$' + ' = {:.3f}'.format(Lgam_u_ea_tavg)

            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.05, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
            if index % progress_cad == 0:
                logger.info("Progress: (%i/%i) writes plotted" %(index+1, count))
    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
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

    if args['--pvort'] is not None:
        plotpvortin = eval(args['--pvort'])
    else:
        plotpvortin = False

    if args['--rms'] is not None:
        plotgamrms = eval(args['--rms'])
    else:
        plotgamrms = False

    if args['--max'] is not None:
        plotgammax = eval(args['--max'])
    else:
        plotgammax = False

    if args['--ea'] is not None:
        plotgamea = eval(args['--ea'])
    else:
        plotgamea = False

    if plotgamrms or plotgammax or plotgamea:
        plotgamin = True
    else:
        plotgamin = False

    options_in = [plotpvortin, plotgamin, plotgamrms, plotgammax, plotgamea]

    if plotgamin:
        # To load in processed_vortex_scalars file
        Nphi, Nr = 768, 384 #640, 320 #768, 384 #512, 256#1024, 512 
        nu = 2e-4 #2e-4 #5e-4 #1e-3 #2e-4 #1e-4 #8e-5 #2e-4 #5e-5
        gamma = 675 #85 #1920 #240 #30 #0
        k_force = 20 #20 #10 #20 #70 #35 #20 #50

        eps = 1 
        alpha = 1e-2

        ring = 0 
        restart_evolved = False #False #True

        #output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
        #output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
        #output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

        output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
        output_suffix += '_eps_{:.1e}'.format(eps)
        output_suffix += '_alpha_{:.1e}'.format(alpha)
        output_suffix += '_ring_{:d}'.format(ring)
        output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
        output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

        processed_vortex_scalars = np.load('../jupiter-process/processed_vortex_scalars_' + output_suffix + '.npy', allow_pickle = True)[()]

        if plotgamrms:
            Lgams_u_rms = processed_vortex_scalars['Lgamma']['data_Lgamma_u_rms']
            Lgam_u_rms_tavg = processed_vortex_scalars['Lgamma']['data_Lgamma_u_rms_tavg'][0]
        if plotgammax:
            Lgams_u_max = processed_vortex_scalars['Lgamma']['data_Lgamma_u_max']
            Lgam_u_max_tavg = processed_vortex_scalars['Lgamma']['data_Lgamma_u_max_tavg'][0]
        if plotgamea:
            Lgam_u_ea_tavg = processed_vortex_scalars['Lgamma']['data_Lgamma_u_ea_tavg'] # constant

    post.visit_writes(args['<files>'], main, output=output_path, options=options_in)


"""
Plot disk outputs from simulations with fixed parameters. (File naming conventions on other types of runs may need additional accomodations)

Usage:
    plot_disk.py <files>... [--output=<dir> --pvort=<bool> --tracking=<bool> --circle=<bool> --radius=<float>]

Options:
    --output=<dir>      output directory [default: ./frames]
    --pvort=<bool>      whether to plot pvort [default: False]
    --tracking=<bool>   whether to plot processed tracking locations [default: False]
    
    --circle=<bool>     whether to plot a circle to highlight a particular radius [default: False]
    --radius=<float>    radius at which to plot circle [default: None]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

# Configure the logging format to include the time
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
def main(filename, start, count, output, options):
    """Save plot of specified tasks for given range of analysis writes."""

    plotpvort, plottracking, plotcircle, plotradius = options

    if plotpvort:
        tasks = ['vort', 'pvort']
    else:
        tasks = ['vort']

    cmap = plt.cm.RdBu_r
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    dpi = 600 #200
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
                dset_phis = np.array(dset.dims[1][0])
                
                # main plot
                paxes, caxes = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)

                if plottracking and (index in ws):
                    x_loc = r_locs[index - ws[0]] * np.cos(phi_locs[index - ws[0]])
                    y_loc = r_locs[index - ws[0]] * np.sin(phi_locs[index - ws[0]])
                    paxes.scatter(x_loc, y_loc, marker = 'o', ec = 'magenta', fc = 'none', s = 35)

                # overlay circle
                if plotcircle:
                    circ_x = plotradius * np.cos(dset_phis)
                    circ_y = plotradius * np.sin(dset_phis)
                    paxes.plot(circ_x, circ_y, color = "purple", lw = 0.5)
                    #paxes.plot(circ_x, circ_y, color = "purple", linestyle = "dotted", label = r'$r = \langle L_{\gamma} \rangle$')
                    #paxes.legend(loc = 'lower left', fontsize = 6)
                
                paxes.axis('off')
                caxes.cla()
                caxes.axis('off')
            
            # Add time title
            title = title_func(file['scales/sim_time'][index])
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

    plotpvort = eval(args['--pvort'])
    plottracking = eval(args['--tracking'])
    plotcircle = eval(args['--circle'])

    # string parsing if we want to load in additional files from processing
    if plottracking or plotcircle:
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

        file_str = args['<files>'][0]
        output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
        Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
        Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
        alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
        gamma_str = output_suffix.split('gam_')[1].split('_')[0]
        eps_str = output_suffix.split('eps_')[1].split('_')[0]
        nu_str = output_suffix.split('nu_')[1].split('_')[0]
        alpha_read = str_to_float(alpha_str)
        gamma_read = str_to_float(gamma_str)
        eps_read = str_to_float(eps_str)
        nu_read = str_to_float(nu_str)
        alpha_vals = np.array((1e-2, 3.3e-2))
        gamma_vals = np.array((0, 30, 85, 240, 400, 675, 1200, 1920, 2500, 3200))
        eps_vals = np.array([3.3e-1, 1.0, 2.0])
        nu_vals = np.array([5e-5, 2e-4])
        alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
        gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
        eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
        nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]

    if plottracking:
        #processed_tracking = np.load('../jupiter-process/processed_tracking_nu_5em05_gam_2d5ep03_kf_4d0ep01_Nphi_1280_Nr_640_eps_1d0ep00_alpha_3d3em02_ring_0_restart_evolved_0_tau_mod_1_seed_31415926_safety_1d0em01_timestepper_SBDF2_ncc_cutoff1d0em06_implicit_0.npy', allow_pickle = True)[()]
        processed_tracking = np.load('../jupiter-process/processed_tracking_' + output_suffix + '.npy', allow_pickle = True)[()]

        r_locs = processed_tracking['r_locs']
        phi_locs = processed_tracking['phi_locs']
        ws = processed_tracking['ws']

    if plotcircle:
        if eval(args['--radius']) is None:
            # default to plotting Lgamma
            processed_scalars = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle = True)[()] 
            ts = processed_scalars['t']
            if ts[-1] - ts[0] >= 2/alpha:
                t_dur = 1/alpha
            else:
                t_dur = 0.5 * (ts[-1] - ts[0])
            startidx = np.where(ts >= ts[-1] - t_dur)[0][0]
            u_rms = np.sqrt(2 * np.mean(processed_scalars['KE'][startidx:]))
            L_gamma = (u_rms / gamma)**(1/3)
            plotradius = L_gamma
            logger.info('default plot radius will be at L_gamma = %e' %(plotradius))
        else:
            plotradius = float(args['--radius'])
    else:
        plotradius = None
    
    options_in = [plotpvort, plottracking, plotcircle, plotradius]

    post.visit_writes(args['<files>'], main, output=output_path, options=options_in)

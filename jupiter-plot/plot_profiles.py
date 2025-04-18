"""
Plot profiles from joint analysis files.

Usage:
    plot_profiles.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib import transforms
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    tasks = ['w0', 'drw0']
    titles = {}
    titles['w0'] = r'$\omega_0$'
    titles['drw0'] = r'$\partial_r \omega_0$'

    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(2, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

     # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]

                print(index, n, task, dset.shape)
                pbbox = transforms.Bbox.from_bounds(0.2, 0.2, 0.8, 0.8)
                to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
                pbbox = pbbox.transformed(to_axes_bbox)
                paxes = axes.figure.add_axes(pbbox)
                axes.axis("off")

                xdata = np.array(dset.dims[2][0])
                ydata = dset[index, 0, :]

                paxes.plot(xdata, ydata, color = "black", linewidth = 1)
                paxes.set_title(titles[task])
                paxes.set_xlabel(dset.dims[2].label)

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
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

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

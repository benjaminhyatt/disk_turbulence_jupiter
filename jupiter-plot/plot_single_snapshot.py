"""
Plot highres disk snapshot.

Usage:
    plot_single_snapshot.py <files>... [--index=<int>]

Options:
    --index=<int>
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from dedalus.extras import plot_tools

from docopt import docopt

args = docopt(__doc__)

filename = args['<files>'][0]
file = h5py.File(filename, mode='r+') # read and write

if args['--index'] is not None:
    indexin = int(args['--index'])
else:
    indexin = 1

# plot scale
plot_scale = 4

# params
Nphi, Nr = 768, 384 #640, 320 #768, 384 #512, 256#1024, 512 
nu = 2e-4 #8e-5 #2e-4 #5e-4 #1e-3 #2e-4 #1e-4 #8e-5 #2e-4 #5e-5
gamma = 675 #85 #1920 #240 #30 #0
k_force = 20 #20 #10 #20 #70 #35 #20 #50
eps = 1
alpha = 1e-2
ring = 0
restart_evolved = False #False #True

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

dtype = np.float64
index = indexin
task = 'vort'    
cmap = plt.cm.PuOr_r
dpi = 3000
func = lambda phi, r, data: (r*np.cos(phi), r*np.sin(phi), data)
dset = file['tasks'][task]
#try:
#    dset_scaled = file.create_dataset(task + '_scaled', (1, int(plot_scale * Nphi), int(plot_scale * Nr)), maxshape = (1, int(plot_scale * Nphi), int(plot_scale * Nr)), dtype = dtype)
#except:
#    del file[task + '_scaled']
#    dset_scaled = file.create_dataset(task + '_scaled', (1, int(plot_scale * Nphi), int(plot_scale * Nr)), maxshape = (1, int(plot_scale * Nphi), int(plot_scale * Nr)), dtype = dtype)

dealias = 3/2
#coords = d3.PolarCoordinates('phi', 'r')
#dist = d3.Distributor(coords, dtype=dtype)
#disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
#om = dist.Field(name='om', bases=disk)
#om['g'] = dset[index]
#om.change_scales(plot_scale)
#dset_scaled[:] = om['g']

savename_func = lambda write: 'snapshot_' + output_suffix + '_write_{:06}.png'.format(write)

n = 0
nrows, ncols = 1, 1
image = plot_tools.Box(1, 1)
pad = plot_tools.Frame(0.05, 0, 0, 0)
margin = plot_tools.Frame(0.15, 0.2, 0.01, 0.01)
scale = 3

# Create figure
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
fig = mfig.figure

# Build subfigure axes
i, j = divmod(n, ncols)
axes = mfig.add_axes(i, j, [0, 0, 1, 1])

# main plot
#paxes, caxes = plot_tools.plot_bot_3d(om, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)
#paxes, caxes = plot_tools.plot_bot_3d(dset_scaled, 0, 0, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)
paxes, caxes = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)
paxes.axis('off')
#caxes.cla()
#caxes.axis('off')
caxes.set_xlabel(None) 

# αdd title
title = r"$\omega_z$"
title_height = 1 - 0.3 * mfig.margin.top / mfig.fig.y
fig.suptitle(title, x=0.48, y=title_height, ha='left', fontsize = 14)
# Save figure
savename = savename_func(file['scales/write_number'][index])
fig.savefig(str(savename), dpi=dpi)
fig.clear()

plt.close(fig)

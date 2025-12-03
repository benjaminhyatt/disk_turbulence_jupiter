"""
Make frames of azimuthally-integrated spectra at steady-state

Usage:
    plot_spectra_zb_steady.py <files>... [--output=<dir>]

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

Nphi, Nr = 512, 256
nu = 2e-4
gamma = 85 #0 #1920 #0 
k_force = 20
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

filename = '../jupiter-process/processed_spectra_zb_dini_' + output_suffix + '.npy'

trunc_data = True
trunc_scale = 128 #sig/2pi approx Nr/2

dpi = 200

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 17
plt.rcParams['figure.dpi'] = dpi

t_mar, b_mar, l_mar, r_mar = (0.1, 0.15, 0.15, 0.05)
#golden_mean = (np.sqrt(5) - 1.) / 2.
#h_plot, w_plot = (1., 1. / golden_mean)
h_plot, w_plot = (1., 1.)
h_pad = b_mar
h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

fig_width = 4.8
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

grey_out = False #True
if grey_out:
    colors = ['grey', 'grey', 'grey']
else:
    colors = ['#7570b3', '#1b9e77', '#d95f02']

#ymin = 10**(np.floor(np.log10(np.min(f[tasks[-1]][1:,:]))) + 1)
#ymax = 10**(np.ceil(np.log10(np.max(f[tasks[-1]][1:,:]))))
ymin = 1e-9
ymax = 1e1

ax1 = fig.add_axes([left1, bottom1, width1, height1])
ax1.set_yscale('log')
ax1.set_xscale('log')

xdata = f['centers'] / (2 * np.pi)

for p, task in enumerate(tasks):
    ydata = f[task + '_tavg'][:]
    ydata_tavg = np.copy(ydata[np.abs(ydata) > 1e-14])
    xdata_tavg = np.copy(xdata[np.abs(ydata) > 1e-14])
    if trunc_data:
        ydata_tavg = np.copy(ydata_tavg[xdata_tavg <= trunc_scale])
        xdata_tavg = np.copy(xdata_tavg[xdata_tavg <= trunc_scale])
    if task != 'keBn':
        ax1.plot(xdata_tavg, ydata_tavg, linestyle = 'solid', linewidth = 4.5, label = labels[task], color = colors[p])
    else:
        ax1.plot(xdata_tavg, ydata_tavg, linestyle = 'dotted', linewidth = 4.5, label = labels[task], color = colors[p])
#if plot_tavg:
#    ydata = f[task_tavg][:]
#    ax1.plot(xdata, ydata, markersize = 2.5, linestyle = "dashed", dashes = (5, 6), color = "black")

ax1.set_xlabel(r'$\sigma / 2\pi$')
#ax1.set_ylabel(labels[task])
ax1.set_ylim(ymin, ymax)

if not grey_out:
    ax1.legend(loc = "lower left")


# Add time title
#title = 'Kinetic energy'
#title_height = 1 - 0.25 * t_mar
#fig.suptitle(title, x=0.44, y=title_height, ha='left')
# Save figure
if grey_out:
    fig.savefig('spectra_zb_dini_steady_grey_out_' + output_suffix + '.png', dpi=dpi)
else:
    fig.savefig('spectra_zb_dini_steady_' + output_suffix + '.png', dpi=dpi)
fig.clear()

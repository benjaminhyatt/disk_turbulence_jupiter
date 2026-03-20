"""
Usage:
    plot_spectra_zb_steady.py <file>... [options]

Options:    
    --dini=<bool>               True: uses a Dini expansion (Bessel with H=1 Robin bc); False: uses a Bessel expansion (Dirichlet bc) [default: True]
    --mmaxplot=<int>            Integer of the highest value of m to include in plot [default: 3]
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

from docopt import docopt
args = docopt(__doc__)

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

print("args read in")
print(args)

mmaxplot = int(args['--mmaxplot'])

file_str = args['<file>'][0]
dini = eval(args['--dini'])
if dini:
    output_prefix = 'processed_spectra_zb_mbin_dini'
else:
    output_prefix = 'processed_spectra_zb_mbin_std'
output_suffix = file_str.split(output_prefix + '_')[1].split('.')[0].split('/')[0] #[:-1] 

Nr = int(output_suffix.split('Nr_')[1].split('_')[0])

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
alpha_vals = np.array((1e-2, 3.3e-2, 1e-1))
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]

trunc_data = True
trunc_scale = Nr/np.pi

dpi = 200

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 17
plt.rcParams['figure.dpi'] = dpi

t_mar, b_mar, l_mar, r_mar = (0.1, 0.20, 0.20, 0.05)
#t_mar, b_mar, l_mar, r_mar = (0.1, 0.15, 0.2, 0.05)
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

f = np.load(file_str, allow_pickle = True)[()]
msplot = np.arange(0, mmaxplot+1)
tasks = []
labels = []
for mplot in msplot:
    tasks.append('keBmn')
    labels.append(f'$E_{{m={mplot}}}$')

print(msplot)
print(f['keBmn_tavg'].shape)
print(f['keBmn_tavg'][0, :])
print(f['keBmn_tavg'][1, :])
print(f['keBmn_tavg'][2, :])

#tasks = ['keBn_zonal', 'keBn_nz', 'keBn']
#labels = {}
#labels['keBn'] = r'$K_{\rm tot}$'
#labels['enBn'] = r'$Z_{\rm tot}$'
#labels['fluxBn'] = r'$\Pi_{\rm tot}$'
#labels['keBn_zonal'] = r'$K_{m = 0}$'
#labels['enBn_zonal'] = r'$Z_{m = 0}$'
#labels['fluxBn_zonal'] = r'$\Pi_{m = 0}$'
#labels['keBn_nz'] = r'$K_{m \ne 0}$'
#labels['enBn_nz'] = r'$Z_{m \ne 0}$'
#labels['fluxBn_nz'] = r'$\Pi_{m \ne 0}$'
#colors = ['#7570b3', '#1b9e77', '#d95f02']

ymin = 5e-10
ymax = 5e1


ax1 = fig.add_axes([left1, bottom1, width1, height1])
ax1.set_yscale('log')
ax1.set_xscale('log')

xdata = f['centers'] / (2 * np.pi)

for mplot in msplot:
    task = tasks[mplot]
    label = labels[mplot]
    ydata = f[task + '_tavg'][mplot, :]
    ydata_tavg = np.copy(ydata[np.abs(ydata) > 1e-14])
    xdata_tavg = np.copy(xdata[np.abs(ydata) > 1e-14])
    if trunc_data:
        ydata_tavg = np.copy(ydata_tavg[xdata_tavg <= trunc_scale])
        xdata_tavg = np.copy(xdata_tavg[xdata_tavg <= trunc_scale])
    ax1.plot(xdata_tavg, ydata_tavg, linestyle = 'solid', linewidth = 2.75, label = label)

#Cth = 6.0
#print(np.sum(f['keB_tavg']))
#KE_tavg = np.sum(f['keB_tavg']) / np.pi
#print(KE_tavg)
#eps_value = 2 * alpha * KE_tavg * np.pi
#print(eps_value)
#ax1.plot(xdata_tavg, Cth*(eps_value)**(2/3)*(xdata_tavg*2*np.pi)**(-5/3), color = "black", ls = "dashed")

ax1.set_xlabel(r'$k / 2\pi$')
ax1.set_ylim(ymin, ymax)

ax1.legend(loc = "lower left")
if dini:
    fig.savefig('ke_spectra_zb_mbin_dini_steady_' + output_suffix + '.png', dpi=dpi)
    print('saving figure: ke_spectra_zb_mbin_dini_steady_' + output_suffix + '.png')
else:
    fig.savefig('ke_spectra_zb_mbin_std_steady_' + output_suffix + '.png', dpi=dpi)
    print('saving figure: ke_spectra_zb_mbin_std_steady_' + output_suffix + '.png')
fig.clear()


#######
ymin = 5e-12
ymax = 5e1

ax1 = fig.add_axes([left1, bottom1, width1, height1])
ax1.set_yscale('log')
ax1.set_xscale('log')

xdata = f['centers'] / (2 * np.pi)

for mplot in msplot:
    task = tasks[mplot]
    label = labels[mplot]
    ydata = f[task + '_tavg'][mplot, :]
    ydata_tavg = np.copy(ydata[np.abs(ydata) > 1e-14])
    xdata_tavg = np.copy(xdata[np.abs(ydata) > 1e-14])
    if trunc_data:
        ydata_tavg = np.copy(ydata_tavg[xdata_tavg <= trunc_scale])
        xdata_tavg = np.copy(xdata_tavg[xdata_tavg <= trunc_scale])
    ax1.plot(xdata_tavg, ydata_tavg, linestyle = 'solid', linewidth = 2.75, label = label)
ax1.set_xlabel(r'$k / 2\pi$')
ax1.set_ylim(ymin, ymax)

ax1.legend(loc = "lower left")

if dini:
    fig.savefig('en_spectra_zb_mbin_dini_steady_' + output_suffix + '.png', dpi=dpi)
    print('saving figure: en_spectra_zb_mbin_dini_steady_' + output_suffix + '.png')
else:
    fig.savefig('en_spectra_zb_mbin_std_steady_' + output_suffix + '.png', dpi=dpi)
    print('saving figure: en_spectra_zb_mbin_std_steady_' + output_suffix + '.png')
fig.clear()

import numpy as np
import matplotlib.pyplot as plt
import h5py

#import matplotlib
import cmasher as cmr 
cmap = cmr.get_sub_cmap('berlin', 0.0, 1.0)

gammas = [85, 240, 400, 675, 1920]

Nphi, Nr = 512, 256
nu = 2e-4
k_force = 20
eps = 1
alpha = 1e-2
ring = 0
restart_evolved = False 

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix = 'nu_{:.0e}'.format(nu) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

processed = np.load('../jupiter-process/processed_averages_' + output_suffix + '.npy', allow_pickle=True)[()]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 600 
fontsize = 10

##### make 5-panel figure #####
t_mar, b_mar, l_mar, r_mar = (0.12, 0.1, 0.1, 0.12)

h_plot, w_plot = (1., 1.)
w_pad = 0.05 * w_plot
h_pad = 0.05 * h_plot
h_cbar = h_pad

h_total = t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot + b_mar
w_total = l_mar + w_plot + w_pad + w_plot + w_pad + w_plot + r_mar

fig_width = 5.5
scale = fig_width/w_total

fig = plt.figure(figsize = (scale * w_total, scale * h_total))


##### construct axs #####
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax1 = fig.add_axes([left, bottom, width, height])

left = (l_mar + w_plot + w_pad) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax2 = fig.add_axes([left, bottom, width, height])

left = (l_mar + w_plot + w_pad + w_plot + w_pad) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax3 = fig.add_axes([left, bottom, width, height])

left = (l_mar + 0.5 * (w_plot + w_pad)) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax4 = fig.add_axes([left, bottom, width, height])

left = (l_mar + 1.5 * (w_plot + w_pad)) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax5 = fig.add_axes([left, bottom, width, height])

axs = [ax1, ax2, ax3, ax4, ax5]

##### construct cax #####
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_cbar) / h_total
width = (3 * w_plot + w_pad) / w_total
height = h_cbar / h_total
cax = fig.add_axes([left, bottom, width, height])

vmax_all = 0.
for gamma in gammas:
    Z = processed[gamma]['Z']
    vmax_all = np.max((vmax_all, Z.max()))

for gam, ax in enumerate(axs):
    gamma = gammas[gam]
    print("maxing ax for", gamma)

    X = processed[gamma]['X']
    Y = processed[gamma]['Y']
    Z = processed[gamma]['Z']
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap, vmin = 0., vmax = vmax_all)
    #ax.set_xticklabels(())
    #ax.set_yticklabels(())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.set_axis_off()

    perim = np.linspace(0, 2*np.pi, int(2e3))
    rad = 1 - 5e-3
    ax.plot(rad * np.cos(perim), rad * np.sin(perim), linewidth = 0.15, color = "black")

    ha = 'right'
    xx = 1.1
    ax.text(xx, 1.0, r'$\gamma = $' + '%d' %(gamma), ha = ha, va = 'top', transform=ax.transAxes)

cbar = fig.colorbar(mesh, cax = cax, orientation = 'horizontal', label = 'Time-averaged cyclone distribution', ticks = [])
cbar.ax.xaxis.set_label_position('top')

plt.savefig("heat_maps_3b2_" + output_suffix + '.png')
plt.clf()


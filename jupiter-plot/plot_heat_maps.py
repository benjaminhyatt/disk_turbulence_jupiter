import numpy as np
import matplotlib.pyplot as plt
import h5py

def gaussian(x, a, mu, sig):
    return a * np.exp(-0.5*((x-mu)/sig)**2) / np.sqrt(2*np.pi*(sig**2))

def gaussian_origin(x, a, sig):
    return gaussian(x, a, 0., sig)

gammas_fig_1 = [85, 240, 400, 675, 1920]
gammas_fig_2 = [240, 400, 675, 1920]

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

##### make 1-panel figure #####
t_mar, b_mar, l_mar, r_mar = (0.1, 0.2, 0.2, 0.1)

golden_mean = (np.sqrt(5) - 1.) / 2.
h_plot, w_plot = (1., 1. / golden_mean)
#w_pad = 0.05 * w_plot
#h_pad = 0.05 * h_plot
h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

fig_width = 5.5
scale = fig_width/w_total

fig = plt.figure(figsize = (scale * w_total, scale * h_total))

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax = fig.add_axes([left, bottom, width, height])

#colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
colors = ['#ffffb2', '#fecc5c', 'black', '#f03b20', '#bd0026']

ylimtop = 0.

for gam, gamma in enumerate(gammas_fig_1):
    print("plotting azimuthally-averaged profile for", gamma)

    xplot = processed[gamma]['r_deal'][0, :]
    yplot = processed[gamma]['Z_m0'][0, :]

    ax.fill_between(xplot, yplot, alpha = 0.2, color = colors[gam])
    ax.plot(xplot, yplot, color = colors[gam], label = r'$\gamma = $' + '%d' %(gamma))

    ax.set_xlabel(r'$r$')
    ax.set_ylabel('Time-averaged cyclone distribution')
    ax.set_yticks(())

    ylimtop = np.max((ylimtop, yplot.max()))

ax.legend(loc = "upper right")

for gam, gamma in enumerate(gammas_fig_1):
    xplot = processed[gamma]['r_deal'][0, :]
    mean = processed[gamma]['Z_m0_expect']
    var = np.abs(processed[gamma]['Z_m0_var'])
    amp = processed[gamma]['Z_m0_amp']
    print(gamma, mean, var)
    print(gamma, processed[gamma]['L_gam'])
    if (gamma != 1920) and (gamma != 400):
        mean_height = gaussian(mean, amp, mean, var)
        varL_height = gaussian(mean - var, amp, mean, var)
        varR_height = gaussian(mean + var, amp, mean, var)
        ax.vlines(mean, 0, mean_height, color = colors[gam], linestyle = "dashed")
        ax.vlines(mean - var, 0, varL_height, color = colors[gam], linestyle = "dotted", linewidth = 3)
    else:
        varR_height = gaussian_origin(mean + var, amp, var) #gaussian(mean + var, amp, mean, var) # also works
    ax.vlines(mean + var, 0, varR_height, color = colors[gam], linestyle = "dotted", linewidth = 3)

    #L_gam = processed[gamma]['L_gam']
    #yplot = processed[gamma]['Z_m0']
    #ax.vlines(L_gam, 0, yplot.max(), color = colors[gam], linestyle = "dotted")
    #ylimtop = np.max((ylimtop, yplot.max()))
#ax.set_xlim(0., 1.)
ax.set_xlim(0., 0.6)
ax.set_ylim(0., 1.05 * ylimtop)

plt.savefig("heat_maps_1d_" + output_suffix + '.png')
plt.clf()

##### make 4-panel figure #####
t_mar, b_mar, l_mar, r_mar = (0.1, 0.1, 0.1, 0.1)

h_plot, w_plot = (1., 1.)
w_pad = 0.05 * w_plot
h_pad = 0.05 * h_plot
h_cbar = h_pad

h_total = t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot + b_mar
w_total = l_mar + w_plot + w_pad + w_plot + r_mar

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

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax3 = fig.add_axes([left, bottom, width, height])

left = (l_mar + w_plot + w_pad) / w_total
bottom = 1 - (t_mar + h_pad + h_cbar + h_plot + h_pad + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total
ax4 = fig.add_axes([left, bottom, width, height])

axs = [ax1, ax2, ax3, ax4]

##### construct cax #####
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_cbar) / h_total
width = (2. * w_plot + w_pad) / w_total
height = h_cbar / h_total
cax = fig.add_axes([left, bottom, width, height])

vmax_all = 0.
for gamma in gammas_fig_2:
    Z = processed[gamma]['Z']
    vmax_all = np.max((vmax_all, Z.max()))

for gam, ax in enumerate(axs):
    gamma = gammas_fig_2[gam]
    print("maxing ax for", gamma)

    X = processed[gamma]['X']
    Y = processed[gamma]['Y']
    Z = processed[gamma]['Z']
    mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='gist_heat_r', vmin = 0., vmax = vmax_all)
    #ax.set_xticklabels(())
    #ax.set_yticklabels(())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.set_axis_off()

    perim = np.linspace(0, 2*np.pi, int(2e3))
    rad = 1 - 5e-3
    ax.plot(rad * np.cos(perim), rad * np.sin(perim), linewidth = 0.15, color = "black")

    ha = 'right'
    xx = 1.0
    ax.text(xx, 1.0, r'$\gamma = $' + '%d' %(gamma), ha = ha, va = 'top', transform=ax.transAxes)

cbar = fig.colorbar(mesh, cax = cax, orientation = 'horizontal', label = 'Time-averaged cyclone distribution', ticks = [])
cbar.ax.xaxis.set_label_position('top')

plt.savefig("heat_maps_" + output_suffix + '.png')
plt.clf()


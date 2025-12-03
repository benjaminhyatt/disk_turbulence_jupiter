import numpy as np
import matplotlib.pyplot as plt
import h5py

def gaussian(x, a, mu, sig):
    return a * np.exp(-0.5*((x-mu)/sig)**2) / np.sqrt(2*np.pi*(sig**2))

def gaussian_origin(x, a, sig):
    return gaussian(x, a, 0., sig)

gammas = [85, 240, 400, 675, 1920]

Nphi, Nr = 512, 256
nu = 2e-4
k_force = 20
eps = 1
alpha = 1e-2
ring = 0
restart_evolved = False 

use_std = False # True

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix = 'nu_{:.0e}'.format(nu) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

if use_std:
    processed = np.load('../jupiter-process/processed_averages_std_' + output_suffix + '.npy', allow_pickle=True)[()]
else:
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

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot) / h_total
width = w_plot / w_total
height = h_plot / h_total

ylimtop = 0.
for gamma in gammas:
    yplot = processed[gamma]['Z_m0'][0, :]
    ylimtop = np.max((ylimtop, np.max(np.abs(yplot))))

#colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
colors = ['#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']

for fign in range(len(gammas)):
    fig = plt.figure(figsize = (scale * w_total, scale * h_total))
    ax = fig.add_axes([left, bottom, width, height])    

    for gam, gamma in enumerate(gammas[0:fign+1]):
        print("plotting azimuthally-averaged profile for", gamma)

        xplot = processed[gamma]['r_deal'][0, :]
        yplot = processed[gamma]['Z_m0'][0, :]

        ax.fill_between(xplot, yplot, alpha = 0.2, color = colors[gam])
        ax.plot(xplot, yplot, color = colors[gam], label = r'$\gamma = $' + '%d' %(gamma))

        ax.set_xlabel(r'$r$')
        if use_std:
            ax.set_ylabel('Time-averaged zonal vorticity')
        else:
            ax.set_ylabel('Time-averaged cyclone distribution')
        ax.set_yticks(())

    ax.legend(loc = "upper right")

    for gam, gamma in enumerate(gammas[0:fign+1]):
        xplot = processed[gamma]['r_deal'][0, :]
        yplot = processed[gamma]['Z_m0'][0, :]
        if not use_std:
            mean = processed[gamma]['Z_m0_expect']
            var = np.abs(processed[gamma]['Z_m0_var'])
            amp = processed[gamma]['Z_m0_amp']
            #print(gamma, mean, var)
            if (gamma != 1920) and (gamma != 400):
                mean_height = gaussian(mean, amp, mean, var)
                varL_height = gaussian(mean - var, amp, mean, var)
                varR_height = gaussian(mean + var, amp, mean, var)
                ax.vlines(mean, 0, mean_height, color = colors[gam], linestyle = "dashed")
                ax.vlines(mean - var, 0, varL_height, color = colors[gam], linestyle = "dotted")
                ax.vlines(mean + var, 0, varR_height, color = colors[gam], linestyle = "dotted")
            else:
                print("skipping")
                #varR_height = gaussian_origin(mean + var, amp, var) #gaussian(mean + var, amp, mean, var) # also works
                #nearvarR_height = yplot[np.argmin(np.abs(xplot - (mean + var)))]
                #print(np.argmin(np.abs(xplot - (mean + var))), nearvarR_height)
                #varR_height = np.min((nearvarR_height, varR_height))
            #ax.vlines(mean + var, 0, varR_height, color = colors[gam], linestyle = "dotted")
    
    if use_std:
        ax.set_xlim(0., 1.0)
        ax.set_ylim(-1.05 * ylimtop, 1.05 * ylimtop)
    else:
        #ax.set_xlim(0., 0.6)
        ax.set_xlim(0., 1.0)
        ax.set_ylim(0., 1.05 * ylimtop)

    if use_std:
        plt.savefig("heat_maps_1d_std_" + str(fign) + '_' + output_suffix + '.png')
    else:
        plt.savefig("heat_maps_1d_" + str(fign) + '_' + output_suffix + '.png')
    plt.clf()


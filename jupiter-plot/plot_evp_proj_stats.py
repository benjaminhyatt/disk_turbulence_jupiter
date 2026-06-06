"""
Usage:
    plot_evp_proj_stats.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the file being loaded [default: processed_rossby_projection]
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from docopt import docopt
args = docopt(__doc__)
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

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

file_str = args['<file>'][0]
output_prefix = args['--output']
output_suffix = file_str.split(output_prefix + '_')[1].split('.')[0].split('/')[0] #[:-1] 

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
alpha_vals = np.array((1e-2, 3.3e-2, 1e-1))
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]

kf_str = output_suffix.split('kf_')[1].split('_')[0]
kf_read = str_to_float(kf_str)
kf_vals = np.array((10, 20, 30, 40, 80))

k_force = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]
k_f = k_force / 2 
k_f *= 2 * np.pi

gam_str = output_suffix.split('gam_')[1].split('_')[0]
gam_read = str_to_float(gam_str)
gam_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
gamma = gam_vals[np.argmin(np.abs(gam_vals - gam_read))]

### loading output from processed_rossby_proj ###
processed = np.load(file_str, allow_pickle=True)[()]

idxs_include = list(processed['idxs_include'])

#idxs_include = idxs_include[:3]

print(processed['evals_re'])

evals_re = processed['evals_re'][idxs_include]
evals_im = processed['evals_im'][idxs_include]
drifts = processed['drifts'][idxs_include]
projdot = processed['projdot']
projdot_stddev = processed['projdot_stats'][idxs_include, 1]
projl2 = processed['projl2']
ivpl2 = processed['ivpl2']
evpl2 = processed['evpl2']
t = processed['tw']

projdot_c = processed['projdot_c']
projdot_s = processed['projdot_s']

#print(evals_re)
#print(evals_re[idxs_include])

#projl2_fits = processed['projl2_fits']

### plotting defaults ###

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

### criteria ###
stddev_thresh = 3e-4 # don't plot modes whose stddev are below this value
re2im_thresh = 1e1 # only let modes through if their real part is larger than their imaginary part by this factor
im_accept = 1e0 # make an acception if their imaginary parts are smaller than this value
opts = [stddev_thresh, re2im_thresh, im_accept]
def check_criteria(stddev, re2im, im, opts):
    if stddev >= stddev_thresh:
        if re2im >= re2im_thresh:
            return True
        elif im <= im_accept:
            return True
        else:
            return False
    else:
        return False

### plots ###

### main projection results ###

plt.figure(figsize=(8,4))
for i, idx in enumerate(idxs_include):
    if check_criteria(projdot_stddev[i], np.abs(evals_re[idx]/evals_im[idx]), np.abs(evals_im[idx]), opts):
        plt.plot(t, projl2[i, :], linewidth=1, label = r"$\lambda =$" + f"{evals_re[idx]:.2f}" + r"$+ i$" + f"{evals_im[idx]:.2f}")
        
plt.ylabel(r'$\|{\rm proj}_{\psi_{\rm EVP}}\left(\psi_{\rm IVP}(t)\right)\|$')
plt.xlabel(r'$t$ (short test window)')
plt.title(r'Norms of simulation Fourier mode $\psi_{\rm IVP}$ projected onto eigenmode $\psi_{\rm EVP}$')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.35, 1.))
plt.yscale('symlog', linthresh = 0.5*stddev_thresh)
plt.tight_layout()

plt.savefig('rossby_proj_' + output_suffix + '.pdf')
print('saving figure: ' + 'rossby_proj_' + output_suffix + '.pdf')
plt.close()

### looking at fits ###
for i, idx in enumerate(idxs_include):
    if check_criteria(projdot_stddev[i], np.abs(evals_re[idx]/evals_im[idx]), np.abs(evals_im[idx]), opts):
        fit = processed['projl2_fit'][i, :]
        fit_params = processed['projl2_fit_params'][i, :]
        rel_err = processed['projl2_rel_err'][i, :]
        print(fit_params)
        #if i == 0 or (np.mean(np.abs(rel_err)) < 2):
            #plt.plot(t, rel_err, linewidth=1, label = r"Fit: $\lambda =$" +f"{fit_params[1]:.2f}, " + r"$A =$" + f"{fit_params[0]:.2f}, " + ", EVP: $\lambda =$" + f"{evals_re[idx]:.2f}" + r"$+ i$" + f"{evals_im[idx]:.2f}")
        plt.plot(t, projl2[i, :], linewidth=1, label = r"Projection, " + "EVP: $\lambda =$" + f"{evals_re[idx]:.2f}" + r"$+ i$" + f"{evals_im[idx]:.2f}")
        plt.plot(t, fit, linewidth=1, label = r"Fit: $\lambda =$" +f"{fit_params[1]:.2f}, " + r"$A =$" + f"{fit_params[0]:.2f}, " + ", EVP: $\lambda =$" + f"{evals_re[idx]:.2f}" + r"$+ i$" + f"{evals_im[idx]:.2f}")

#plt.ylabel('Relative error')
plt.xlabel(r'$t$ (short test window)')
#plt.title('Relative error of sine fit')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.35, 1.))
plt.yscale('symlog', linthresh = 0.5*stddev_thresh)
plt.tight_layout()

plt.savefig('rossby_proj_rel_err_' + output_suffix + '.pdf')
print('saving figure: ' + 'rossby_proj_rel_err_' + output_suffix + '.pdf')
plt.close()


omega_tracking = {}
omega_tracking[gamma] = {}
try:
    omega_tracking_prev = np.load('omega_tracking.npy', allow_pickle=True)[()]
    omega_tracking.update(omega_tracking_prev)
except:
    print('no file found')
omega_tracking[gamma]['omega_re_evp'] = []
omega_tracking[gamma]['omega_im_evp'] = []
omega_tracking[gamma]['omega_re_fit'] = []

for i, idx, in enumerate(idxs_include):    
    if np.sum(processed['projl2_fit_params'][i, :]) > 0: # check that the fitting procedure didn't fail
        omega_tracking[gamma]['omega_re_evp'].append(evals_re[idx])
        omega_tracking[gamma]['omega_im_evp'].append(evals_im[idx])
        omega_tracking[gamma]['omega_re_fit'].append(processed['projl2_fit_params'][i, 1])
        # can add more here in the future

print(omega_tracking.keys())
np.save('omega_tracking.npy', omega_tracking)

processed_tracking = np.load('../jupiter-process/processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_1_tau_mod_1_seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy', allow_pickle=True)[()]
tw_t = processed_tracking['tw']
print(t, tw_t)

r_locs = processed_tracking['r_locs']
phi_locs = processed_tracking['phi_locs']

### quick look at r(t) and phi(t) from tracking
plt.figure()
plt.plot(tw_t, r_locs, label = 'r')
plt.plot(tw_t, phi_locs, label = 'phi')
plt.legend()
plt.savefig('r_phi_quick.pdf')

### quick comparison between phi(t) and dominant mode amplitude
plt.figure()
i = 0 # at this point I think things are ordered by amplitude when read-in
plt.plot(t, projl2[i, :]  / np.max(projl2[i, :]), label = 'proj amp (dominant)')
plt.plot(tw_t, phi_locs / np.max(phi_locs), label = 'phi')
plt.legend()
plt.savefig('proj_phi_quick.pdf')





plt.figure(figsize=(8,4))
for i, idx in enumerate(idxs_include[:1]):
    if check_criteria(projdot_stddev[i], np.abs(evals_re[idx]/evals_im[idx]), np.abs(evals_im[idx]), opts):
        plt.plot(t, projdot[i, :], linewidth=2, label = r"projdot $\lambda =$" + f"{evals_re[idx]:.2f}" + r"$+ i$" + f"{evals_im[idx]:.2f}")
        plt.plot(t, projdot_c[i, :] + projdot_s[i, :], linestyle = 'dashed', linewidth=1.5, label = "sum of projdot_c and projdot_s")
        plt.plot(t, projdot_c[i, :], linewidth=0.7, label = "projdot_c")
        plt.plot(t, projdot_s[i, :], linewidth=0.7, label = "projdot_s")
        
#plt.ylabel(r'$\|{\rm proj}_{\psi_{\rm EVP}}\left(\psi_{\rm IVP}(t)\right)\|$')
plt.xlabel(r'$t$ (short test window)')
#plt.title(r'Norms of simulation Fourier mode $\psi_{\rm IVP}$ projected onto eigenmode $\psi_{\rm EVP}$')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.35, 1.))
plt.yscale('symlog', linthresh = 0.5*stddev_thresh)
plt.tight_layout()

plt.savefig('rossby_proj_cs_check_' + output_suffix + '.pdf')
print('saving figure: ' + 'rossby_proj_cs_check_' + output_suffix + '.pdf')
plt.close()








#gammas = list(omega_tracking.keys())
#gammas = np.sort(gammas)
#omega_re_evps = []
#omega_re_fits = []

#gammas_plot = np.array((30, 85, 240, 400, 675, 950, 1200, 1920))
#for gam in gammas_plot:
#    #try:
#    #    maxi = 0
#    #except: 
#    #    maxi = max_idx
#    maxi = 0
#    omega_re_evps.append(omega_tracking[gam]['omega_re_evp'][maxi])
#    omega_re_fits.append(omega_tracking[gam]['omega_re_fit'][maxi])
#plt.figure()
#plt.plot(gammas_plot, omega_re_evps, '-o', label='linear theory')
#plt.plot(gammas_plot, omega_re_fits, '-o', label='simulation fit')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'oscillation frequency $\omega$')
#plt.xscale('log')
#plt.yscale('log')
#plt.title(r'$m=1, k=k_{m, 0}$')
#plt.legend()
#plt.savefig('omega_tracking_' + str(maxi) + '.pdf')








#print(fft_fwd)
#print(np.abs(fft_fwd))
#print(fft_freq)

#maxf = np.max(np.abs(fft_fwd))#.real))
#freqmaxf = np.abs(fft_freq[np.abs(fft_fwd) == maxf])#.real) == maxf][0])
#print(freqmaxf)
#which_freq = np.argmin(np.abs(freqmaxf - evals_re[idxs_include[i]]))
#print(freqmaxf[which_freq])
#freqmaxf = freqmaxf[which_freq]

#p0 = (np.max(projl2[i, :]), freqmaxf, 0.) 
#bds = ([0.95*p0[0], 0.1*freqmaxf, -np.inf], [1.05*p0[0], 10*freqmaxf, np.inf])

#print(maxf, freqmaxf, bds)

#cf = curve_fit(sine_fit, t, projl2[i, :], p0, bounds=bds) 
#fit_params = cf[0]
#print(fit_params)

#fit = sine_fit(t, fit_params[0], fit_params[1], fit_params[2])
#rel_err = np.abs(projl2[i, :] - fit) / fit

#plt.figure()
#plt.plot(t, projl2[i, :])
#plt.plot(t, fit)
#plt.savefig('testingfit.png')










# below is old?

### plots ###
#n_label = 5
#n_nolabel = len(idxs_include) - n_label
#cmap_grays = cm.get_cmap('Grays')
#colors_label_1 = ['blue', 'red', 'green', 'purple', 'brown']
#colors_label_2 = ['cyan', 'orange', 'lightgreen', 'pink', 'yellow']
#colors_nolabel = cmap_grays(np.linspace(0.2, 1.0, n_nolabel))

#plt.figure(figsize=(8,4))
#for i, idx in enumerate(idxs_include[:n_label]):
#    #plt.plot(t, projl2[idx, :]*(evpl2[idx, 0]/ivpl2[idx, :]), color = colors_label_1[i], linewidth=1, label = r"$\lambda =$" + f"{evals_re[i]:.2f}" + r"$+ i$" + f"{evals_im[i]:.2f}")
    #plt.plot(t, projl2[idx, :]*(np.sqrt(evpl2[idx, 0])/np.sqrt(ivpl2[idx, :])), color = colors_label_1[i], linewidth=1, label = r"$\lambda =$" + f"{evals_re[i]:.2f}" + r"$+ i$" + f"{evals_im[i]:.2f}")
#    #plt.plot(t, projdot[idx, :]*(evpl2[idx, 0]/ivpl2[idx, :]), color = colors_label_2[i], linewidth=1, ls='dashed')
#    #plt.plot(t, projdot[idx, :]*(np.sqrt(evpl2[idx, 0])/np.sqrt(ivpl2[idx, :])), color = colors_label_2[i], linewidth=1, ls='dashed')
#    ###plt.axhline(y=projl2_mean[i]*evpl2[i, 0]/ivpl2_mean[i], color = colors_label_1[i], linestyle='dotted', linewidth=0.5)
#    plt.plot(t, projl2[idx, :], color = colors_label_1[i], linewidth=1, label = r"$\lambda =$" + f"{evals_re[i]:.2f}" + r"$+ i$" + f"{evals_im[i]:.2f}")
#    plt.plot(t, projdot[idx, :], color = colors_label_2[i], linewidth=1, ls='dashed')
#for i, idx in enumerate(idxs_include[n_label:]):
#    #plt.plot(t, projl2[idx, :]*(np.sqrt(evpl2[idx, :])/np.sqrt(ivpl2[idx, :])), color = colors_nolabel[i, :], linewidth=0.05)
#    plt.plot(t, projl2[idx, :], color = colors_nolabel[i, :], linewidth=0.05)
#plt.ylabel(r'$\|{\rm proj}_{\psi_{\rm EVP}}\left(\psi_{\rm IVP}(t)\right)\|$')
#plt.xlabel(r'$t$ (short test window)')
#plt.title(r'Norms of simulation vorticity $\omega_{\rm IVP}$ projected onto eigenmode $\omega_{\rm EVP}$')
#plt.legend(loc = 'upper right', bbox_to_anchor=(1.35, 1.))
#plt.yscale('symlog', linthresh = 5e-2)
##plt.ylim(-1, 1)
#plt.tight_layout()
##plt.legend(loc = (1.05, 0.05))

#plt.savefig('rossby_proj_stats_' + output_suffix + '.pdf')
#print('saving figure: ' + 'rossby_proj_stats_' + output_suffix + '.pdf')
#plt.close()

#plt.figure(figsize=(8,4))
#for i, idx in enumerate(idxs_include[:n_label]):
#    plt.plot(t, ivpl2[idx, :]/evpl2[idx], color = colors_label_1[i], linewidth=1, label = r"$\lambda =$" + f"{evals_re[i]:.2f}" + r"$+ i$" + f"{evals_im[i]:.2f}")
#for i, idx in enumerate(idxs_include[n_label:]):
#    plt.plot(t, ivpl2[idx, :]/evpl2[idx], color = colors_nolabel[i, :], linewidth=0.05)
#plt.ylabel(r'$\|\omega_{\rm IVP}\| / \|\omega_{\rm EVP}\|$')
#plt.xlabel(r'$t$ (short test window)')
#plt.title(r'Rations between the norms of simulation vorticity $\omega_{\rm IVP}$ and eigenmode $\omega_{\rm EVP}$')
##plt.legend(loc = 'upper right', bbox_to_anchor=(1.35, 1.))
#plt.yscale('symlog', linthresh = 5e-2)
#plt.ylim(-1, 1)
#plt.tight_layout()
##plt.legend(loc = (1.05, 0.05))

#plt.savefig('rossby_proj_stats_ratios_' + output_suffix + '.pdf')
#print('saving figure: ' + 'rossby_proj_stats_ratios_' + output_suffix + '.pdf')




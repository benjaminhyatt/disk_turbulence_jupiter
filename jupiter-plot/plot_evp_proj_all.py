import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

### load summary data ###
omega_tracking = np.load('omega_tracking.npy', allow_pickle=True)[()]
    
### plots ###

### this one plots the linear theory prediction for whichever mode the IVP finds to be the most dominant

omega_re_evps = []
omega_re_fits = []
#gammas_plot = np.array((30, 85, 240, 400, 675, 950, 1200, 1920)) #2500
gammas_plot = np.array((30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200)) # just working on 1920 (and 3200)

mode_i = 0 # should be ordered by dominance in amplitude
for gam in gammas_plot:
    omega_re_evps.append(omega_tracking[gam]['omega_re_evp'][mode_i])
    omega_re_fits.append(omega_tracking[gam]['omega_re_fit'][mode_i])

omega_re_evps = np.array(omega_re_evps)
omega_re_fits = np.array(omega_re_fits)

plt.figure()
plt.plot(gammas_plot, omega_re_evps, '-o', label='linear theory')
plt.plot(gammas_plot, omega_re_fits, '-o', label='simulation fit')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'oscillation frequency $\omega$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Frequenies of dominant mode for a given $\gamma$')
plt.legend()
plt.savefig('omega_tracking_ivp_dom_evp.pdf')
plt.close()

### briefly, same idea, but in the form of absolute signed error
plt.figure()
plt.plot(gammas_plot, omega_re_fits - omega_re_evps, '-o')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\omega_{\rm fit} - \omega_{\rm pred}$')
plt.xscale('log')
#plt.yscale('log')
plt.title(r'Frequenies of dominant mode for a given $\gamma$')
plt.savefig('omega_tracking_ivp_dom_evp_err.pdf')
plt.close()

plt.figure()
plt.plot(gammas_plot, (omega_re_fits - omega_re_evps)/omega_re_evps, '-o')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$(\omega_{\rm fit} - \omega_{\rm pred}) / \omega_{\rm pred}$')
plt.xscale('log')
#plt.yscale('log')
plt.title(r'Frequenies of dominant mode for a given $\gamma$')
plt.savefig('omega_tracking_ivp_dom_evp_rel_err.pdf')
plt.close()




### this one plots three branches of linear theory predictions (not taken from evp, this is just straight up analytical dispersion relation -- more convenient indexing wise)
def rossby_dispersion(m, g, k, a, n):
    re_part = m * g / k**2 # positive oscillation frequency
    im_part = a + n * k**2 # positive decay rate
    return re_part, im_part

from scipy.special import jn_zeros

m = 1
ks = jn_zeros(m, 3)
alpha = 1e-2
nu = 2e-4

omega_re_disp = np.zeros((len(ks), len(gammas_plot)))
omega_im_disp = np.zeros((len(ks), len(gammas_plot)))
for i, k in enumerate(ks):
    for j, g in enumerate(gammas_plot):
        omega_re_disp[i, j], omega_im_disp[i, j] = rossby_dispersion(m, g, k, alpha, nu)

plt.figure()
for i, k in enumerate(ks):
    plt.plot(gammas_plot, omega_re_disp[i, :], '--', label='linear theory, ' + r'$k =$' + f'{k:.4f}')

mode_i = 0 # should be ordered by dominance in amplitude
omega_re_fits = []
for gam in gammas_plot:
    omega_re_fits.append(omega_tracking[gam]['omega_re_fit'][mode_i])
plt.plot(gammas_plot, omega_re_fits, '-o', label='simulation fit')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'oscillation frequency $\omega$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'Frequenies of dominant mode for a given $\gamma$')
plt.legend()
plt.savefig('omega_tracking_ivp_dom_disp.pdf')

# all ivp measurements that we have, not just the dominant ones -- unless the fit was unsuccessful

plt.figure()
for i, k in enumerate(ks):
    plt.plot(gammas_plot, omega_re_disp[i, :], '--', label='linear theory, ' + r'$k =$' + f'{k:.4f}')

for mode_i in range(3): # I set this up in a way that the plot script that runs before this might omit a mode that failed, so going up to range(10) can introduce errors
    omega_re_fits_i = []
    gammas_plot_i = []
    for j, gam in enumerate(gammas_plot):
        if omega_tracking[gam]['omega_re_fit'][mode_i] != 0:
            omega_re_fits_i.append(omega_tracking[gam]['omega_re_fit'][mode_i])
            gammas_plot_i.append(float(gammas_plot[j]))
        else:
            omega_re_fits_i.append(np.nan)
    plt.scatter(gammas_plot_i, omega_re_fits_i, marker = 'x', color = 'black')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'oscillation frequency $\omega$')
plt.xscale('log')
plt.yscale('log')
plt.title(r'all modes that were smooth enough to fit (currently biased to first singular value)')
plt.legend()
plt.savefig('omega_tracking_ivp_all_disp.pdf')

#





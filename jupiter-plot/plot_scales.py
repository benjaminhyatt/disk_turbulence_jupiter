"""
Specify all processed_scalars files to load from

Usage:
    plot_scales.py <files>...
"""

import numpy as np
import matplotlib.pyplot as plt
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

gamma_vals = np.array([0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200])
kf_vals = np.array([20, 30, 40]) # working with an old convention: 2*(periods per R^(-1))/(2*pi), i.e., 1/pi x the correct wavenumber of the forcing scale

print("args read in")
print(args)
nfiles = len(args['<files>'])

# separate files by forcing scale
group_ref = np.zeros(nfiles, dtype = np.int32)
n_in_group = np.zeros(kf_vals.shape[0])
for n, file_str in enumerate(args['<files>']):
    output_suffix = file_str.split('../jupiter-process/processed_scalars_')[1].split('.')[0].split('/')[0] #[:-1] 
    kf_str = output_suffix.split('kf_')[1].split('_')[0]
    kf_read = str_to_float(kf_str)
    kf = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]
    
    kf_val_idx = np.where(kf == kf_vals)[0][0]    
    group_ref[n] = kf_val_idx 
    n_in_group[kf_val_idx] += 1

# to store values
store = {}
items = ['rossby_freq_kf', 'turnover_freq_kf', 'shear_freq_kf', 'k_tr', 'k_rh', 'k_edsh', 'R_gam', 'R_shi', 'turnover_freq_kf_nu0', 'shear_freq_kf_nu0', 'k_tr_nu0', 'k_rh_nu0', 'k_edsh_nu0', 'R_gam_nu0', 'R_shi_nu0', 'k_edsh_rev', 'R_gam_rev', 'R_shi_rev']
for kf_val in kf_vals:
    store[kf_val] = {}
    store[kf_val]['gamma'] = np.array([])
    for item in items:
        store[kf_val][item] = np.array([])
    store[kf_val]['rossby_freqs'] = np.array([])
    store[kf_val]['turnover_freqs'] = np.array([])
    store[kf_val]['shear_freqs'] = np.array([])
    store[kf_val]['k_trs'] = np.array([])
    store[kf_val]['k_rhs'] = np.array([])
    store[kf_val]['k_edshs'] = np.array([])
    store[kf_val]['R_gams'] = np.array([])
    store[kf_val]['R_shis'] = np.array([])

    store[kf_val]['turnover_freqs_nu0'] = np.array([])
    store[kf_val]['shear_freqs_nu0'] = np.array([])
    store[kf_val]['k_trs_nu0'] = np.array([])
    store[kf_val]['k_rhs_nu0'] = np.array([])
    store[kf_val]['k_edshs_nu0'] = np.array([])
    store[kf_val]['R_gams_nu0'] = np.array([])
    store[kf_val]['R_shis_nu0'] = np.array([])

    store[kf_val]['k_edshs_rev'] = np.array([])
    store[kf_val]['R_gams_rev'] = np.array([])
    store[kf_val]['R_shis_rev'] = np.array([])

#gammas_kf1 = np.array([])
#rossby_freqs_kf1 = np.array([])
#turnover_freqs_kf1 = np.array([])
#shear_freqs_kf1 = np.array([])
#k_trs_kf1 = np.array([])
#k_rhs_kf1 = np.array([])
#k_edshs_kf1 = np.array([])
##k_omshs_kf1 = np.array([])
#R_gams_kf1 = np.array([])
#R_shis_kf1 = np.array([])
#turnover_freqs_nu0_kf1 = np.array([])
#shear_freqs_nu0_kf1 = np.array([])
#k_trs_nu0_kf1 = np.array([])
#k_rhs_nu0_kf1 = np.array([])
#k_edshs_nu0_kf1 = np.array([])
##k_omshs_nu0_kf1 = np.array([])
#R_gams_nu0_kf1 = np.array([])
#R_shis_nu0_kf1 = np.array([])
#k_edshs_rev_kf1 = np.array([])
#R_gams_rev_kf1 = np.array([])
#R_shis_rev_kf1 = np.array([])

#gammas_kf2 = np.array([])
#rossby_freqs_kf2 = np.array([])
#turnover_freqs_kf2 = np.array([])
#shear_freqs_kf2 = np.array([])
#k_trs_kf2 = np.array([])
#k_rhs_kf2 = np.array([])
#k_edshs_kf2 = np.array([])
##k_omshs_kf2 = np.array([])
#R_gams_kf2 = np.array([])
#R_shis_kf2 = np.array([])
#turnover_freqs_nu0_kf2 = np.array([])
#shear_freqs_nu0_kf2 = np.array([])
#k_trs_nu0_kf2 = np.array([])
#k_rhs_nu0_kf2 = np.array([])
#k_edshs_nu0_kf2 = np.array([])
##k_omshs_nu0_kf2 = np.array([])
#R_gams_nu0_kf2 = np.array([])
#R_shis_nu0_kf2 = np.array([])
#R_gams_rev_kf2 = np.array([])
#k_edshs_rev_kf2 = np.array([])
#R_shis_rev_kf2 = np.array([])

# extract processed values
for n, file_str in enumerate(args['<files>']):
    # load processed scalars
    processed = np.load(file_str, allow_pickle=True)[()]
    output_suffix = file_str.split('../jupiter-process/processed_scalars_')[1].split('.')[0].split('/')[0]    
    gam_str = output_suffix.split('gam_')[1].split('_')[0]
    gam_read = str_to_float(gam_str)
    gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gam_read))] 
    
    kf_val = kf_vals[group_ref[n]]
    store[kf_val]['gamma'] = np.append(store[kf_val]['gamma'], gamma)
    for item in items:
        store[kf_val][item] = np.append(store[kf_val][item], processed[item])
    
    #if group_ref[n] == 1:
    #    gammas_kf1 = np.append(gammas_kf1, gamma)
    #    rossby_freqs_kf1 = np.append(rossby_freqs_kf1, processed['rossby_freq_kf'])
    #    turnover_freqs_kf1 = np.append(turnover_freqs_kf1, processed['turnover_freq_kf'])
    #    shear_freqs_kf1 = np.append(shear_freqs_kf1, processed['shear_freq_kf'])
    #    k_trs_kf1 = np.append(k_trs_kf1, processed['k_tr'])
    #    k_rhs_kf1 = np.append(k_rhs_kf1, processed['k_rh'])
    #    k_edshs_kf1 = np.append(k_edshs_kf1, processed['k_edsh'])
    #    #k_omshs_kf1 = np.append(k_omshs_kf1, processed['k_omsh'])
    #    R_gams_kf1 = np.append(R_gams_kf1, processed['R_gam'])
    #    R_shis_kf1 = np.append(R_shis_kf1, processed['R_shi'])
    #    turnover_freqs_nu0_kf1 = np.append(turnover_freqs_nu0_kf1, processed['turnover_freq_kf_nu0'])
    #    shear_freqs_nu0_kf1 = np.append(shear_freqs_nu0_kf1, processed['shear_freq_kf_nu0'])
    #    k_trs_nu0_kf1 = np.append(k_trs_nu0_kf1, processed['k_tr_nu0'])
    #    k_rhs_nu0_kf1 = np.append(k_rhs_nu0_kf1, processed['k_rh_nu0'])
    #    k_edshs_nu0_kf1 = np.append(k_edshs_nu0_kf1, processed['k_edsh_nu0'])
    #    #k_omshs_nu0_kf1 = np.append(k_omshs_nu0_kf1, processed['k_omsh_nu0'])
    #    R_gams_nu0_kf1 = np.append(R_gams_nu0_kf1, processed['R_gam_nu0'])
    #    R_shis_nu0_kf1 = np.append(R_shis_nu0_kf1, processed['R_shi_nu0'])
    #    R_gams_rev_kf1 = np.append(R_gams_rev_kf1, processed['R_gam_rev'])
    #    k_edshs_rev_kf1 = np.append(k_edshs_rev_kf1, processed['k_edsh_rev'])
    #    R_shis_rev_kf1 = np.append(R_shis_rev_kf1, processed['R_shi_rev'])
    #elif group_ref[n] == 2:
    #    gammas_kf2 = np.append(gammas_kf2, gamma)
    #    rossby_freqs_kf2 = np.append(rossby_freqs_kf2, processed['rossby_freq_kf'])
    #    turnover_freqs_kf2 = np.append(turnover_freqs_kf2, processed['turnover_freq_kf'])
    #    shear_freqs_kf2 = np.append(shear_freqs_kf2, processed['shear_freq_kf'])
    #    k_trs_kf2 = np.append(k_trs_kf2, processed['k_tr'])
    #    k_rhs_kf2 = np.append(k_rhs_kf2, processed['k_rh'])
    #    k_edshs_kf2 = np.append(k_edshs_kf2, processed['k_edsh'])
    #    #k_omshs_kf2 = np.append(k_omshs_kf2, processed['k_omsh'])
    #    R_gams_kf2 = np.append(R_gams_kf2, processed['R_gam'])
    #    R_shis_kf2 = np.append(R_shis_kf2, processed['R_shi'])
    #    turnover_freqs_nu0_kf2 = np.append(turnover_freqs_nu0_kf2, processed['turnover_freq_kf_nu0'])
    #    shear_freqs_nu0_kf2 = np.append(shear_freqs_nu0_kf2, processed['shear_freq_kf_nu0'])
    #    k_trs_nu0_kf2 = np.append(k_trs_nu0_kf2, processed['k_tr_nu0'])
    #    k_rhs_nu0_kf2 = np.append(k_rhs_nu0_kf2, processed['k_rh_nu0'])
    #    k_edshs_nu0_kf2 = np.append(k_edshs_nu0_kf2, processed['k_edsh_nu0'])
    #    #k_omshs_nu0_kf2 = np.append(k_omshs_nu0_kf2, processed['k_omsh_nu0'])
    #    R_gams_nu0_kf2 = np.append(R_gams_nu0_kf2, processed['R_gam_nu0'])
    #    R_shis_nu0_kf2 = np.append(R_shis_nu0_kf2, processed['R_shi_nu0'])
    #    R_gams_rev_kf2 = np.append(R_gams_rev_kf2, processed['R_gam_rev'])
    #    k_edshs_rev_kf2 = np.append(k_edshs_rev_kf2, processed['k_edsh_rev'])
    #    R_shis_rev_kf2 = np.append(R_shis_rev_kf2, processed['R_shi_rev'])

# sort in order of increasing gamma
for kf_val in kf_vals:
    gam_sort = np.argsort(store[kf_val]['gamma'])
    for item in items:
        store[kf_val][item] = store[kf_val][item][gam_sort]
    store[kf_val]['gamma'] = store[kf_val]['gamma'][gam_sort]

#gam_sort_kf1 = np.argsort(gammas_kf1)
#gam_sort_kf2 = np.argsort(gammas_kf2)

#rossby_freqs_kf1 = rossby_freqs_kf1[gam_sort_kf1]
#turnover_freqs_kf1 = turnover_freqs_kf1[gam_sort_kf1]
#shear_freqs_kf1 = shear_freqs_kf1[gam_sort_kf1]
#k_trs_kf1 = k_trs_kf1[gam_sort_kf1]
#k_rhs_kf1 = k_rhs_kf1[gam_sort_kf1]
#R_gams_kf1 = R_gams_kf1[gam_sort_kf1]

#turnover_freqs_nu0_kf1 = turnover_freqs_nu0_kf1[gam_sort_kf1]
#shear_freqs_nu0_kf1 = shear_freqs_nu0_kf1[gam_sort_kf1]
#k_trs_nu0_kf1 = k_trs_nu0_kf1[gam_sort_kf1]
#k_rhs_nu0_kf1 = k_rhs_nu0_kf1[gam_sort_kf1]
#R_gams_nu0_kf1 = R_gams_nu0_kf1[gam_sort_kf1]

#rossby_freqs_kf2 = rossby_freqs_kf2[gam_sort_kf2]
#turnover_freqs_kf2 = turnover_freqs_kf2[gam_sort_kf2]
#shear_freqs_kf2 = shear_freqs_kf2[gam_sort_kf2]
#k_trs_kf2 = k_trs_kf2[gam_sort_kf2]
#k_rhs_kf2 = k_rhs_kf2[gam_sort_kf2]
#R_gams_kf2 = R_gams_kf2[gam_sort_kf2]

#turnover_freqs_nu0_kf2 = turnover_freqs_nu0_kf2[gam_sort_kf2]
#shear_freqs_nu0_kf2 = shear_freqs_nu0_kf2[gam_sort_kf2]
#k_trs_nu0_kf2 = k_trs_nu0_kf2[gam_sort_kf2]
#k_rhs_nu0_kf2 = k_rhs_nu0_kf2[gam_sort_kf2]
#R_gams_nu0_kf2 = R_gams_nu0_kf2[gam_sort_kf2]

#gammas_kf1 = gammas_kf1[gam_sort_kf1]
#gammas_kf2 = gammas_kf2[gam_sort_kf2]


# Plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

# Plot 1: eddy turnover frequencies and Rossby wave (m = 1) frequencies
#plt.figure()
#plt.plot(gammas_kf1, shear_freqs_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, shear_freqs_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, turnover_freqs_kf1, color = 'blue', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\alpha u_{rms}^2)^{1/3} k_f^{2/3}$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, turnover_freqs_kf2, color = 'orange', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\alpha u_{rms}^2)^{1/3} k_f^{2/3}$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, rossby_freqs_kf1, color = 'blue', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby} = \gamma/k_f^2$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, rossby_freqs_kf2, color = 'orange', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby} = \gamma/k_f^2$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'Frequency ($T^{-1}$)')
#plt.legend(loc = 'lower left')
#plt.title(r'Time scales estimated at $k = k_f$')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('time_scales.pdf')
#print('saving figure: ' + 'time_scales.pdf')

# Plot 2: transitional wavenumbers and Rhines (i.e., L_gamma^(-1) based on Siegelman et al 2022) wavenumbers
#plt.figure()
#plt.plot(gammas_kf1, k_trs_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\alpha u_{rms}^2)^{-1/8}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_trs_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\alpha u_{rms}^2)^{-1/8}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (7/gammas_kf1[0]**(3/8)) * gammas_kf1**(3/8), color = 'gray', linestyle = 'dashed', label = r'$\propto \gamma^{3/8}$')

#plt.plot(gammas_kf1, k_rhs_kf1, color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_rhs_kf2, color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (3/gammas_kf1[0]**(1/3)) * gammas_kf1**(1/3), color = 'black', linestyle = 'dashed', label = r'$\propto \gamma^{1/3}$')

#plt.ylabel(r'Wavenumber ($L^{-1}$)')
#plt.xlabel(r'$\gamma$')
#plt.legend(loc = 'upper left')
#plt.title('Length scale (wavenumber) estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('length_scales.pdf')
#print('saving figure: ' + 'length_scales.pdf')

# Plot 3: Zonostrophy indices (based on Sukoriansky et al 2007)
#plt.figure()
#plt.plot(gammas_kf1, R_gams_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, R_gams_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$k_f = 20 \times 2\pi$')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'$R_{\gamma} = k_{\rm tr} / k_{\gamma}$ (dimensionless)')
#plt.legend(loc = 'upper left')
#plt.title('Zonostrophic index estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('zon_scales.pdf')
#print('saving figure: ' + 'zon_scales.pdf')

##### Redo with fully-inviscid estimates #####

# Plot 1: eddy turnover frequencies and Rossby wave (m = 1) frequencies
#plt.figure()
#plt.plot(gammas_kf1, shear_freqs_nu0_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = \sqrt{\epsilon/(\pi\alpha)} k_f$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, shear_freqs_nu0_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = \sqrt{\epsilon/(\pi\alpha)} k_f$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, turnover_freqs_nu0_kf1, color = 'blue', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3}k_f^{2/3}$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, turnover_freqs_nu0_kf2, color = 'orange', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3}k_f^{2/3}$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, rossby_freqs_kf1, color = 'blue', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby} = \gamma/k_f^2$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, rossby_freqs_kf2, color = 'orange', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby} = \gamma/k_f^2$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'Frequency ($T^{-1}$)')
#plt.legend(loc = 'lower left')
#plt.title(r'Time scales estimated at $k = k_f$')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('time_scales_nu0.pdf')
#print('saving figure: ' + 'time_scales_nu0.pdf')

# Plot 2: transitional wavenumbers and Rhines (i.e., L_gamma^(-1) based on Siegelman et al 2022) wavenumbers
#plt.figure()
#plt.plot(gammas_kf1, k_trs_nu0_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_trs_nu0_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (7/gammas_kf1[0]**(3/8)) * gammas_kf1**(3/8), color = 'gray', linestyle = 'dashed', label = r'$\propto \gamma^{3/8}$')

#plt.plot(gammas_kf1, k_rhs_nu0_kf1, color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} \left(\epsilon/(\pi\alpha)\right)^{-1/6}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_rhs_nu0_kf2, color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} \left(\epsilon/(\pi\alpha)\right)^{-1/6}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (3/gammas_kf1[0]**(1/3)) * gammas_kf1**(1/3), color = 'black', linestyle = 'dashed', label = r'$\propto \gamma^{1/3}$')

#plt.ylabel(r'Wavenumber ($L^{-1}$)')
#plt.xlabel(r'$\gamma$')
#plt.legend(loc = 'upper left')
#plt.title('Length scale (wavenumber) estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('length_scales_nu0.pdf')
#print('saving figure: ' + 'length_scales_nu0.pdf')

# Plot 3: Zonostrophy indices (based on Sukoriansky et al 2007)
#plt.figure()
#plt.plot(gammas_kf1, R_gams_nu0_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, R_gams_nu0_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$k_f = 20 \times 2\pi$')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'$R_{\gamma} = k_{\rm tr} / k_{\gamma}$ (dimensionless)')
#plt.legend(loc = 'upper left')
#plt.title('Zonostrophic index estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('zon_scales_nu0.pdf')
#print('saving figure: ' + 'zon_scales_nu0.pdf')

##### Finally, I think this might be the "combination" that we want ##### 

# Plot 1: eddy turnover frequencies and Rossby wave (m = 1) frequencies
plt.figure()

#plt.plot(gammas_kf1, shear_freqs_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, shear_freqs_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, turnover_freqs_nu0_kf1, color = 'blue', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3} k_f^{2/3}$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, turnover_freqs_nu0_kf2, color = 'orange', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3} k_f^{2/3}$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
#plt.plot(gammas_kf1, rossby_freqs_kf1, color = 'blue', marker = '^', markersize = 8, label = r'$\tau_{ro}^{-1} = \gamma/k_f^2$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
#plt.plot(gammas_kf2, rossby_freqs_kf2, color = 'orange', marker = '^', markersize = 8, label = r'$\tau_{ro}^{-1} = \gamma/k_f^2$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['shear_freq_kf'], color = 'blue', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['shear_freq_kf'], color = 'orange', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 15 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['shear_freq_kf'], color = 'green', marker = 's', markersize = 8, label = r'$\tau_{\rm sh}^{-1} = u_{rms} k_f$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2}$)')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['turnover_freq_kf_nu0'], color = 'blue', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3} k_f^{2/3}$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['turnover_freq_kf_nu0'], color = 'orange', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3} k_f^{2/3}$ ($k_f = 15 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['turnover_freq_kf_nu0'], color = 'green', marker = 'o', markersize = 8, label = r'$\tau_{\rm ed}^{-1} = (\epsilon/\pi)^{1/3} k_f^{2/3}$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2    }$)')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['rossby_freq_kf'], color = 'blue', marker = '^', markersize = 8, label = r'$\tau_{ro}^{-1} = \gamma/k_f^2$ ($k_f = 10 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['rossby_freq_kf'], color = 'orange', marker = '^', markersize = 8, label = r'$\tau_{ro}^{-1} = \gamma/k_f^2$ ($k_f = 15 \times 2\pi, \alpha = 10^{-2}$)')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['rossby_freq_kf'], color = 'green', marker = '^', markersize = 8, label = r'$\tau_{ro}^{-1} = \gamma/k_f^2$ ($k_f = 20 \times 2\pi, \alpha = 3.3 \times 10^{-2    }$)')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'Frequency ($T^{-1}$)')
plt.legend(loc = 'lower left')
plt.title(r'Time scales estimated at $k = k_f$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('time_scales_rev.pdf')
print('saving figure: ' + 'time_scales_rev.pdf')

# Plot 2: transitional wavenumbers and Rhines (i.e., L_gamma^(-1) based on Siegelman et al 2022) wavenumbers
#plt.figure()
#plt.plot(gammas_kf1, k_trs_nu0_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_trs_nu0_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 20 \times 2\pi$')

##plt.plot(gammas_kf1, (7/gammas_kf1[0]**(3/8)) * gammas_kf1**(3/8), color = 'gray', linestyle = 'dashed', label = r'$\propto \gamma^{3/8}$')

#plt.plot(gammas_kf1, k_rhs_kf1, color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_rhs_kf2, color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (3/gammas_kf1[0]**(1/3)) * gammas_kf1**(1/3), color = 'black', linestyle = 'dashed', label = r'$\propto \gamma^{1/3}$')

#plt.ylabel(r'Wavenumber ($L^{-1}$)')
#plt.xlabel(r'$\gamma$')
#plt.legend(loc = 'upper left')
#plt.title('Length scale (wavenumber) estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('length_scales_rev.pdf')
#print('saving figure: ' + 'length_scales_rev.pdf')

# Plot 3: Zonostrophy indices (based on Sukoriansky et al 2007)
#plt.figure()
#plt.plot(gammas_kf1, R_gams_rev_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, R_gams_rev_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$k_f = 20 \times 2\pi$')
#plt.xlabel(r'$\gamma$')
#plt.ylabel(r'$R_{\gamma} = k_{\rm tr} / k_{\gamma}$ (dimensionless)')
#plt.legend(loc = 'upper left')
#plt.title('Zonostrophic index estimates')
#plt.xscale('log')
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('zon_scales_rev.pdf')
#print('saving figure: ' + 'zon_scales_rev.pdf')

#### Additional ####
plt.figure()
#plt.plot(gammas_kf1, k_trs_nu0_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_trs_nu0_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 20 \times 2\pi$')
#plt.plot(gammas_kf1, k_rhs_kf1, color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\rm Rh} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_rhs_kf2, color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\rm Rh} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 20 \times 2\pi$')
##plt.plot(gammas_kf1, k_omshs_kf1, color = 'blue', marker = 'o', markersize = 8, label = r'$k_{ro, sh} = u_{\rm rms} / \gamma$, $k_f = 10 \times 2\pi$')
##plt.plot(gammas_kf2, k_omshs_kf2, color = 'orange', marker = 'o', markersize = 8, label = r'$k_{ro, sh} = u_{\rm rms} / \gamma$, $k_f = 20 \times 2\pi$')
#plt.plot(gammas_kf1, k_edshs_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$k_{\rm fric} = u_{\rm rms}^{-3} \epsilon$, $k_f = 10 \times 2\pi$')
#plt.plot(gammas_kf2, k_edshs_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$k_{\rm fric} = u_{\rm rms}^{-3} \epsilon$, $k_f = 20 \times 2\pi$')

plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['k_tr_nu0'],color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['k_tr_nu0'],color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['k_tr_nu0'],color = 'green', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} (\epsilon/\pi)^{-1/8}$, $k_f = 20 \times 2\pi$')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['k_rh'],color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\rm Rh} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['k_rh'],color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\rm Rh} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['k_rh'],color = 'green', marker = 'd', markersize = 8, label = r'$k_{\rm Rh} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 20 \times 2\pi$')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['k_edsh_rev'],color = 'blue', marker = 'o', markersize = 8, label = r'$k_{\rm fric} = u_{\rm rms}^{-3} \epsilon$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['k_edsh_rev'],color = 'orange', marker = 'o', markersize = 8, label = r'$k_{\rm fric} = u_{\rm rms}^{-3} \epsilon$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['k_edsh_rev'],color = 'green', marker = 'o', markersize = 8, label = r'$k_{\rm fric} = u_{\rm rms}^{-3} \epsilon$, $k_f = 20 \times 2\pi$')

plt.ylabel(r'Wavenumber ($L^{-1}$)')
plt.xlabel(r'$\gamma$')
plt.legend(loc = 'upper left')
plt.title('Length scale (wavenumber) estimates')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('length_scales_all.pdf')
print('saving figure: ' + 'length_scales_all.pdf')


plt.figure()
#plt.plot(gammas_kf1, R_gams_rev_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$R_{\gamma} = k_{\rm tr} / k_{\rm Rh}$, ($k_f = 10 \times 2\pi$)')
#plt.plot(gammas_kf2, R_gams_rev_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$R_{\gamma} = k_{\rm tr} / k_{\rm Rh}$, ($k_f = 20 \times 2\pi$)')
#plt.plot(gammas_kf1, R_shis_rev_kf1, color = 'blue', marker = '^', markersize = 8, label = r'$k_{\rm tr}/k_{\rm fric}$, ($k_f = 10 \times 2\pi$)')
#plt.plot(gammas_kf2, R_shis_rev_kf2, color = 'orange', marker = '^', markersize = 8, label = r'$k_{\rm tr}/k_{\rm fric}$, ($k_f = 20 \times 2\pi$)')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['R_gam_rev'],color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} / k_{\rm Rh}$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['R_gam_rev'],color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} / k_{\rm Rh}$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['R_gam_rev'],color = 'green', marker = 's', markersize = 8, label = r'$k_{\rm tr} / k_{\rm Rh}$, $k_f = 20 \times 2\pi$')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['R_shi_rev'],color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\rm tr}/k_{\rm fric}$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['R_shi_rev'],color = 'orange', marker = 'D', markersize = 8, label = r'$_{\rm tr}/k_{\rm fric}$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['R_shi_rev'],color = 'green', marker = 'd', markersize = 8, label = r'$_{\rm tr}/k_{\rm fric}$, $k_f = 20 \times 2\pi$')
plt.plot(store[kf_vals[0]]['gamma'], store[kf_vals[0]]['k_rh'] / store[kf_vals[0]]['k_edsh_rev'],color = 'blue', marker = 'o', markersize = 8, label = r'$k_{\rm Rh}/k_{\rm fric}$, $k_f = 10 \times 2\pi$')
plt.plot(store[kf_vals[1]]['gamma'], store[kf_vals[1]]['k_rh'] / store[kf_vals[1]]['k_edsh_rev'],color = 'orange', marker = 'o', markersize = 8, label = r'$_{\rm Rh}/k_{\rm fric}$, $k_f = 15 \times 2\pi$')
plt.plot(store[kf_vals[2]]['gamma'], store[kf_vals[2]]['k_rh'] / store[kf_vals[2]]['k_edsh_rev'],color = 'green', marker = 'o', markersize = 8, label = r'$_{\rm Rh}/k_{\rm fric}$, $k_f = 20 \times 2\pi$')


plt.xlabel(r'$\gamma$')
plt.legend(loc = 'center left')
plt.title('Dimensionless ratio estimates')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('zon_scales_all.pdf')
print('saving figure: ' + 'zon_scales_all.pdf')


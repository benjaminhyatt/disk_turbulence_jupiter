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
kf_vals = np.array([20, 40]) # working with an old convention: 2*(periods per R^(-1))/(2*pi), i.e., 1/pi x the correct wavenumber of the forcing scale

print("args read in")
print(args)
nfiles = len(args['<files>'])

# separate files by forcing scale
group_ref = np.zeros(nfiles)
for n, file_str in enumerate(args['<files>']):
    output_suffix = file_str.split('../jupiter-process/processed_scalars_')[1].split('.')[0].split('/')[0] #[:-1] 
    kf_str = output_suffix.split('kf_')[1].split('_')[0]
    kf_read = str_to_float(kf_str)
    kf = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]
    if kf == kf_vals[0]:
        group_ref[n] = 1
    elif kf == kf_vals[1]:
        group_ref[n] = 2 
nkf1 = np.sum(group_ref == 1)
nkf2 = np.sum(group_ref == 2)

# to store values
gammas_kf1 = np.array([])
turnover_freqs_kf1 = np.array([])
rossby_freqs_kf1 = np.array([])
k_trs_kf1 = np.array([])
k_rhs_kf1 = np.array([])
R_gams_kf1 = np.array([])

gammas_kf2 = np.array([])
turnover_freqs_kf2 = np.array([])
rossby_freqs_kf2 = np.array([])
k_trs_kf2 = np.array([])
k_rhs_kf2 = np.array([])
R_gams_kf2 = np.array([])

# extract processed values
for n, file_str in enumerate(args['<files>']):
    # load processed scalars
    processed = np.load(file_str, allow_pickle=True)[()]
    output_suffix = file_str.split('../jupiter-process/processed_scalars_')[1].split('.')[0].split('/')[0]    
    gam_str = output_suffix.split('gam_')[1].split('_')[0]
    gam_read = str_to_float(gam_str)
    gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gam_read))] 

    if group_ref[n] == 1:
        gammas_kf1 = np.append(gammas_kf1, gamma)
        turnover_freqs_kf1 = np.append(turnover_freqs_kf1, processed['turnover_freq_kf'])
        rossby_freqs_kf1 = np.append(rossby_freqs_kf1, processed['rossby_freq_kf'])
        k_trs_kf1 = np.append(k_trs_kf1, processed['k_tr'])
        k_rhs_kf1 = np.append(k_rhs_kf1, processed['k_rh'])
        R_gams_kf1 = np.append(R_gams_kf1, processed['R_gam'])
    elif group_ref[n] == 2:
        gammas_kf2 = np.append(gammas_kf2, gamma)
        turnover_freqs_kf2 = np.append(turnover_freqs_kf2, processed['turnover_freq_kf'])
        rossby_freqs_kf2 = np.append(rossby_freqs_kf2, processed['rossby_freq_kf'])
        k_trs_kf2 = np.append(k_trs_kf2, processed['k_tr'])
        k_rhs_kf2 = np.append(k_rhs_kf2, processed['k_rh'])
        R_gams_kf2 = np.append(R_gams_kf2, processed['R_gam'])


# sort in order of increasing gamma
gam_sort_kf1 = np.argsort(gammas_kf1)
gam_sort_kf2 = np.argsort(gammas_kf2)

turnover_freqs_kf1 = turnover_freqs_kf1[gam_sort_kf1]
rossby_freqs_kf1 = rossby_freqs_kf1[gam_sort_kf1]
k_trs_kf1 = k_trs_kf1[gam_sort_kf1]
k_rhs_kf1 = k_rhs_kf1[gam_sort_kf1]
R_gams_kf1 = R_gams_kf1[gam_sort_kf1]

turnover_freqs_kf2 = turnover_freqs_kf2[gam_sort_kf2]
rossby_freqs_kf2 = rossby_freqs_kf2[gam_sort_kf2]
k_trs_kf2 = k_trs_kf2[gam_sort_kf2]
k_rhs_kf2 = k_rhs_kf2[gam_sort_kf2]
R_gams_kf2 = R_gams_kf2[gam_sort_kf2]

gammas_kf1 = gammas_kf1[gam_sort_kf1]
gammas_kf2 = gammas_kf2[gam_sort_kf2]


# Plot settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

# Plot 1: eddy turnover frequencies and Rossby wave (m = 1) frequencies
plt.figure()
plt.plot(gammas_kf1, turnover_freqs_kf1, color = 'blue', marker = 'o', markersize = 8, label = r'$\lambda_{k}$, $k_f = 10 \times 2\pi$')
plt.plot(gammas_kf2, turnover_freqs_kf2, color = 'orange', marker = 'o', markersize = 8, label = r'$\lambda_{k}$, $k_f = 20 \times 2\pi$')
plt.plot(gammas_kf1, rossby_freqs_kf1, color = 'blue', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby}$, $k_f = 10 \times 2\pi$')
plt.plot(gammas_kf2, rossby_freqs_kf2, color = 'orange', marker = '^', markersize = 8, label = r'$\omega_{\rm Rossby}$, $k_f = 20 \times 2\pi$')

plt.xlabel(r'$\gamma$')
plt.ylabel(r'Frequency ($T^{-1}$)')
plt.legend(loc = 'center left')
plt.title(r'Time scales estimated at $k = k_f$')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('time_scales.pdf')
print('saving figure: ' + 'time_scales.pdf')

# Plot 2: transitional wavenumbers and Rhines (i.e., L_gamma^(-1) based on Siegelman et al 2022) wavenumbers
plt.figure()
plt.plot(gammas_kf1, k_trs_kf1, color = 'blue', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} \epsilon_{\alpha}^{-1/8}$, $k_f = 10 \times 2\pi$')
plt.plot(gammas_kf2, k_trs_kf2, color = 'orange', marker = 's', markersize = 8, label = r'$k_{\rm tr} = \gamma^{3/8} \epsilon_{\alpha}^{-1/8}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (7/gammas_kf1[0]**(3/8)) * gammas_kf1**(3/8), color = 'gray', linestyle = 'dashed', label = r'$\propto \gamma^{3/8}$')

plt.plot(gammas_kf1, k_rhs_kf1, color = 'blue', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 10 \times 2\pi$')
plt.plot(gammas_kf2, k_rhs_kf2, color = 'orange', marker = 'd', markersize = 8, label = r'$k_{\gamma} = \gamma^{1/3} u_{\rm rms}^{-1/3}$, $k_f = 20 \times 2\pi$')

#plt.plot(gammas_kf1, (3/gammas_kf1[0]**(1/3)) * gammas_kf1**(1/3), color = 'black', linestyle = 'dashed', label = r'$\propto \gamma^{1/3}$')

plt.ylabel(r'Wavenumber ($L^{-1}$)')
plt.xlabel(r'$\gamma$')
plt.legend(loc = 'upper left')
plt.title('Length scale (wavenumber) estimates')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('length_scales.pdf')
print('saving figure: ' + 'length_scales.pdf')

# Plot 3: Zonostrophy indices (based on Sukoriansky et al 2007)
plt.figure()
plt.plot(gammas_kf1, R_gams_kf1, color = 'blue', marker = '*', markersize = 8, label = r'$k_f = 10 \times 2\pi$')
plt.plot(gammas_kf2, R_gams_kf2, color = 'orange', marker = '*', markersize = 8, label = r'$k_f = 20 \times 2\pi$')
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$R_{\gamma} = k_{\rm tr} / k_{\gamma}$ (dimensionless)')
plt.legend(loc = 'upper left')
plt.title('Zonostrophic index estimates')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('zon_scales.pdf')
print('saving figure: ' + 'zon_scales.pdf')


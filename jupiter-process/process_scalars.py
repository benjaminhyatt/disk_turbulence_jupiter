"""
Usage:
    process_scalars.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the output file [default: processed_scalars]
"""


import numpy as np
import h5py
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

output_prefix = args['--output']

file_str = args['<file>'][0]
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
print(output_suffix)
alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
kf_str = output_suffix.split('kf_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
gamma_read = str_to_float(gamma_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)
kf_read = str_to_float(kf_str)
alpha_vals = np.array((1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 950, 1200, 1920, 2500, 3200))
eps_vals = np.array([3.3e-1, 1.0, 2.0])
nu_vals = np.array([5e-5, 8/90000, 2e-4])
kf_vals = np.array([20, 30, 40]) # working with an old convention: 2*(periods per R^(-1))/(2*pi), i.e., 1/pi x the correct wavenumber of the forcing scale
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]
kf = np.pi * kf_vals[np.argmin(np.abs(kf_vals - kf_read))] # convert to correct value

# load in analysis
f1 = h5py.File(file_str)
t = f1['tasks/KE'].dims[0]['sim_time'][:]
print("t[0],", t[0], ", t[-1],", t[-1])

#Lzu = f1['tasks/Lzu'][:, 0, 0]
KE = f1['tasks/KE'][:, 0, 0]
EN = f1['tasks/EN'][:, 0, 0]
#print(f1['tasks/KE'])
#ENbdry = f1['tasks/ENbdry'][:, 0, 0]
#PA = f1['tasks/PA'][:, 0, 0]
#PAbdry1 = f1['tasks/PAbdry1'][:, 0, 0]
#PAbdry2 = f1['tasks/PAbdry2'][:, 0, 0]
W = f1['tasks/W'][:, 0, 0]
#om0 = f1['tasks/om0'][:, 0, 0]
#om1c = f1['tasks/om1c'][:, 0, 0]
#om1s = f1['tasks/om1s'][:, 0, 0] 
#nu0 = f1['tasks/nu0'][:, 0, 0]
#nu1c = f1['tasks/nu1c'][:, 0, 0]
#nu1s = f1['tasks/nu1s'][:, 0, 0]

tdur = np.max((1.5/alpha, t[-1]/3))
tendidx = -1
tend = t[tendidx]
tstartidx = np.where(t >= tend - tdur)[0][0]
tstart = t[tstartidx]

KE_tavg = np.mean(KE[tstartidx:tendidx])
EN_tavg = np.mean(EN[tstartidx:tendidx])
KE_tavg_expected = ((eps / np.pi) - nu * EN_tavg) / (2 * alpha)
print("KE_tavg", KE_tavg, "EN_tavg", EN_tavg)

# dimensional scales
rossby_freq_def = lambda k, m, gam: m * gam * k**(-2) # Rossby wave frequency
turnover_freq_def = lambda k, ep: ep**(1/3) * k**(2/3) # approximate eddy turnover frequency
shear_freq_def = lambda U, k: U * k # approx shear rate
k_tr_def = lambda ep, gam: ep**(-1/8) * gam**(3/8) # transitional wavenumber (k_{om,ed})
k_rh_def = lambda U, gam: U**(-1/3) * gam**(1/3) # Rhines wavenumber (inverse of L_gamma (Siegelman et al 2022)) (k_{om,sh})
k_edsh_def = lambda U, ep: U**(-3) * ep # transitional wavenumber between eddy time scale and shear time scale (k_{fric})
#k_omsh_def = lambda U, gam: U / gam # transitional wavenumber between rossby time scale and shear time scale (note: = k_rh^(-3)) # note: mistaken in derivation, actually k_rh)

# dimensionless scale(s) 
R_gam_def = lambda k_tr, k_rh: k_tr * k_rh**(-1) # Zonostrophy index (Sukoriansky et al. 2007) in the gamma plane approximation 
R_shi_def = lambda k_tr, k_fric: k_tr * k_fric**(-1) # dimensionless scale from Shi et al. 2025

# estimates
eps_a = 2 * alpha * KE_tavg
urms = np.sqrt(2 * KE_tavg)
rossby_freq_kf = rossby_freq_def(kf, 1, gamma)
turnover_freq_kf = turnover_freq_def(kf, eps_a)
shear_freq_kf = shear_freq_def(urms, kf)
k_tr = k_tr_def(eps_a, gamma)
k_rh = k_rh_def(urms, gamma)
k_edsh = k_edsh_def(urms, eps_a)
#k_omsh = k_omsh_def(urms, gamma)
R_gam = R_gam_def(k_tr, k_rh)
R_shi = R_shi_def(k_tr, k_edsh)

# inviscid estimates
eps_a_nu0 = eps / np.pi
urms_nu0 = np.sqrt(eps / (alpha * np.pi))

turnover_freq_kf_nu0 = turnover_freq_def(kf, eps_a_nu0) 
shear_freq_kf_nu0 = shear_freq_def(urms_nu0, kf)
k_tr_nu0 = k_tr_def(eps_a_nu0, gamma)
k_rh_nu0 = k_rh_def(urms_nu0, gamma)
k_edsh_nu0 = k_edsh_def(urms_nu0, eps_a_nu0)
#k_omsh_nu0 = k_omsh_def(urms_nu0, gamma)
R_gam_nu0 = R_gam_def(k_tr_nu0, k_rh_nu0)
R_shi_nu0 = R_shi_def(k_tr_nu0, k_edsh_nu0)

# revised selection
R_gam_rev = R_gam_def(k_tr_nu0, k_rh)
k_edsh_rev = k_edsh_def(urms, eps_a_nu0)
R_shi_rev = R_shi_def(k_tr_nu0, k_edsh_rev)

# output
processed = {}

processed['t'] = t
#processed['Lzu'] = Lzu
processed['KE'] = KE
processed['EN'] = EN
#processed['ENbdry'] = ENbdry
#processed['PA'] = PA
#processed['PAbdry1'] = PAbdry1
#processed['PAbdry2'] = PAbdry2


#processed['om0'] = om0
processed['W'] = W
#processed['om1c'] = om1c
#processed['om1s'] = om1s

#processed['nu0'] = nu0
#processed['nu1c'] = nu1c
#processed['nu1s'] = nu1s

processed['KE_tavg'] = KE_tavg
processed['EN_tavg'] = EN_tavg
processed['KE_growth_pred'] = (eps/(np.pi)) * t # these are predicted for area-average, i.e., 1/pi * K_tot 
processed['KE_tavg_expected'] = KE_tavg_expected

processed['eps_a'] = eps_a
processed['urms'] = urms
processed['eps_a_nu0'] = eps_a_nu0
processed['urms_nu0'] = urms_nu0

processed['rossby_freq_kf'] = rossby_freq_kf
processed['turnover_freq_kf'] = turnover_freq_kf
processed['shear_freq_kf'] = shear_freq_kf
processed['k_tr'] = k_tr
processed['k_rh'] = k_rh
processed['k_edsh'] = k_edsh
#processed['k_omsh'] = k_omsh
processed['R_gam'] = R_gam
processed['R_shi'] = R_shi
processed['turnover_freq_kf_nu0'] = turnover_freq_kf_nu0
processed['shear_freq_kf_nu0'] = shear_freq_kf_nu0
processed['k_tr_nu0'] = k_tr_nu0
processed['k_rh_nu0'] = k_rh_nu0
processed['k_edsh_nu0'] = k_edsh_nu0
#processed['k_omsh_nu0'] = k_omsh_nu0
processed['R_gam_nu0'] = R_gam_nu0
processed['R_shi_nu0'] = R_shi_nu0

processed['R_gam_rev'] = R_gam_rev
processed['k_edsh_rev'] = k_edsh_rev
processed['R_shi_rev'] = R_shi_rev

print("eps_a", eps_a, "eps_a_nu0", eps_a_nu0)
print("urms", urms, "urms_nu0", urms_nu0)
print("rossby_freq_kf", rossby_freq_kf)
print("turnover_freq_kf", turnover_freq_kf, "nu0", turnover_freq_kf_nu0)
print("shear_freq_kf", shear_freq_kf, "nu0", shear_freq_kf_nu0)
print("k_tr", k_tr, "nu0", k_tr_nu0)
print("k_rh", k_rh, "nu0", k_rh_nu0)
print("k_edsh", k_edsh, "nu0", k_edsh_nu0, "revised", k_edsh_rev)
#print("k_omsh", k_omsh, "nu0", k_omsh_nu0)
print("R_gam", R_gam, "nu0", R_gam_nu0, "revised", R_gam_rev)
print("R_shi", R_shi, "nu0", R_shi_nu0, "revised", R_shi_rev)

print("L_gamma", k_rh**(-1))

w_rms = np.sqrt(EN_tavg)
processed['w_rms'] = w_rms
print("w_rms", w_rms)

print('saving output')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)


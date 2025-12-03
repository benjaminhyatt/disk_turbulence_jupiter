import numpy as np
import matplotlib.pyplot as plt
import h5py

Nphi, Nr = 512, 256
nu = 2e-4
gamma = 0
k_force = 20

alpha = 1e-2
amp = 1

ring = 0

restart_evolved = False #False #True
restart_hyst = False #True
hystn = 7

old = False
if old:
    eps = 2 * amp**2
else:
    eps = amp**2

tau_mod = False
output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
if restart_hyst:
    output_suffix += '_restart_hyst_{:d}'.format(hystn)

processed_0 = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]

tau_mod = True
output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
if restart_hyst:
    output_suffix += '_restart_hyst_{:d}'.format(hystn)

processed_1 = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]


## plots ##
plt.figure()

processed = processed_0
tau_str = ' default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['EN'] / k_force**2, color = 'orange', linestyle = tau_ls, label = r'${\cal E}_{\rm avg} / k_f^2 = \frac{1}{\pi k_f^2}\iint |\omega|^2 dA$' + tau_str)
plt.plot(processed['t'], (processed['EN_tavg'] / k_force**2) * np.ones(processed['t'].shape[0]), linestyle = tau_ls, color = 'red', label = r'$\langle {\cal E}_{\rm avg} \rangle / k_f^2$' + tau_str)

plt.plot(processed['t'], processed['KE'], color = 'blue', linestyle = tau_ls, label = r'${\cal K}_{\rm avg} = \frac{1}{\pi}\iint \frac{1}{2}|u|^2 dA$' + tau_str)
plt.plot(processed['t'], processed['KE_tavg'] * np.ones(processed['t'].shape[0]), linestyle = tau_ls, color = 'black', label = r'$\langle {\cal K}_{\rm avg} \rangle$' + tau_str)

processed = processed_1
tau_str = ' mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['EN'] / k_force**2, color = 'orange', linestyle = tau_ls, label = r'${\cal E}_{\rm avg} / k_f^2 = \frac{1}{\pi k_f^2}\iint |\omega|^2 dA$' + tau_str)
plt.plot(processed['t'], (processed['EN_tavg'] / k_force**2) * np.ones(processed['t'].shape[0]), linestyle = tau_ls, color = 'red', label = r'$\langle {\cal E}_{\rm avg} \rangle / k_f^2$' + tau_str)

plt.plot(processed['t'], processed['KE'], color = 'blue', linestyle = tau_ls, label = r'${\cal K}_{\rm avg} = \frac{1}{\pi}\iint \frac{1}{2}|u|^2 dA$' + tau_str)
plt.plot(processed['t'], processed['KE_tavg'] * np.ones(processed['t'].shape[0]), linestyle = tau_ls, color = 'black', label = r'$\langle {\cal K}_{\rm avg} \rangle$' + tau_str)

#plt.plot(processed['t'], processed['KE_growth_pred'], color ='purple', linestyle = "dotted", label = r'Expected (initial) $\langle \partial_t {\cal K}_{\rm avg} \rangle = (\epsilon / \pi) t$')
#plt.plot(processed['t'], processed['KE_tavg_expected'] * np.ones(processed['t'].shape[0]), linestyle = 'dashdot', color = 'purple', label = r'Expected $\langle {\cal K}_{\rm avg} \rangle = \frac{1}{2\alpha}\left(\frac{\epsilon}{\pi} - \nu {\cal E}_{\rm avg} \right)$')
#plt.plot(processed['t'], 8 * np.pi**2 * processed['KE_growth_pred'], color ='pink', linestyle = "dotted", label = r'Expected (initial) $\langle \partial_t {\cal E}_{\rm avg} \rangle = (8 \epsilon \pi) t$')

plt.xlabel(r'$t$')
plt.ylim(-0.5, 1.1*processed['KE_tavg_expected'])
plt.legend(loc = "lower right", framealpha = 0.9)
plt.tight_layout()
plt.savefig("KE_tau_mod.pdf")



plt.figure()

processed = processed_0
tau_str = 'default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['W'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_0 = np.mean(processed['W'])

processed = processed_1
tau_str = 'mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['W'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_1 = np.mean(processed['W'])

plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.ylabel('net vorticity')
plt.legend()
plt.title("Time averages: default = %e, mod = %e" %(tavg_0, tavg_1))
plt.tight_layout()
plt.savefig("W_tau_mod.pdf")



plt.figure()

processed = processed_0
tau_str = 'default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['Lzu'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_0 = np.mean(processed['Lzu'])

processed = processed_1
tau_str = 'mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['Lzu'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_1 = np.mean(processed['Lzu'])

plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.ylabel('net angular momentum')
plt.legend()
plt.title("Time averages: default = %e, mod = %e" %(tavg_0, tavg_1))
plt.tight_layout()
plt.savefig("Lzu_tau_mod.pdf")



plt.figure()

processed = processed_0
tau_str = 'default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['EN'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_0 = np.mean(processed['EN'])

processed = processed_1
tau_str = 'mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['EN'], linestyle = tau_ls, color = 'blue', label = tau_str)
tavg_1 = np.mean(processed['EN'])

plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.ylabel('Avg enstrophy')
plt.legend()
plt.title("Time averages: default = %e, mod = %e" %(tavg_0, tavg_1))
plt.tight_layout()
plt.savefig("EN_tau_mod.pdf")


plt.figure()

processed = processed_0
tau_str = ' default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['ENbdry'], linestyle = tau_ls, color = 'blue', label ='ENbdry' + tau_str)
plt.plot(processed['t'], processed['PA'], linestyle = tau_ls, color = 'orange', label ='PA' + tau_str)
plt.plot(processed['t'], processed['PAbdry1'], linestyle = tau_ls, color = 'green', label ='PAbdry1' + tau_str)
plt.plot(processed['t'], processed['PAbdry2'], linestyle = tau_ls, color = 'red', label ='PAbdry2' + tau_str)

processed = processed_1
tau_str = ' mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['ENbdry'], linestyle = tau_ls, color = 'blue', label ='ENbdry' + tau_str)
plt.plot(processed['t'], processed['PA'], linestyle = tau_ls, color = 'orange', label ='PA' + tau_str)
plt.plot(processed['t'], processed['PAbdry1'], linestyle = tau_ls, color = 'green', label ='PAbdry1' + tau_str)
plt.plot(processed['t'], processed['PAbdry2'], linestyle = tau_ls, color = 'red', label ='PAbdry2' + tau_str)

plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.yscale('symlog')
plt.legend()
plt.tight_layout()
plt.savefig("bdry_tau_mod.pdf")



plt.figure()

processed = processed_0
tau_str = 'default'
tau_ls = 'solid'
plt.plot(processed['t'], processed['PAbdry1'], linestyle = tau_ls, color = 'green', label = tau_str)
tavg_0 = np.mean(processed['PAbdry1'])

processed = processed_1
tau_str = 'mod'
tau_ls = 'dotted'
plt.plot(processed['t'], processed['PAbdry1'], linestyle = tau_ls, color = 'green', label = tau_str)
tavg_1 = np.mean(processed['PAbdry1'])

plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.legend()
plt.title("Time averages: default = %e, mod = %e" %(tavg_0, tavg_1))
plt.tight_layout()
plt.savefig("PAbdry1_tau_mod.pdf")



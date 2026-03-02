import numpy as np
import matplotlib.pyplot as plt
import h5py

gammai = 4e2
gammaf = 2.4e2
dgamma = 4e1
ke_dt = 1e1
ke_rtol_mm = 2e-1
ke_rtol_lin = 1e-1

seed_in = 31415926 #10101 #31415926 
#analysis_nu_2em04_gami_4d0ep02_gamf_2d4ep02_dgam_4d0ep01_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_tau_mod_1_seed_31415926

tau_mod = True
Nphi, Nr = 512, 256 
nu = 2e-4
k_force = 20
alpha = 1e-2
amp = 1 
ring = 0 
eps = amp**2

output_suffix = 'nu_{:.0e}'.format(nu)
output_suffix += '_gami_{:.1e}'.format(gammai)
output_suffix += '_gamf_{:.1e}'.format(gammaf)
output_suffix += '_dgam_{:.1e}'.format(dgamma)
output_suffix += '_kf_{:.1e}'.format(k_force)
output_suffix += '_Nphi_{:}'.format(Nphi)
output_suffix += '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_seed_{:d}'.format(seed_in)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

processed = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]


#print(processed['KE_tavg_expected'])
#processed['KE_tavg_expected'] = (1/(2*alpha)) * ((2/np.pi) - nu * processed['EN_tavg'])
#print(processed['KE_tavg_expected'])

titlestr = r'$\gamma_i = $' + '{:.1f}'.format(gammai) + r', $k_f = $' +  '{:.1f}'.format(k_force) + r', $\epsilon = $' + '{:.1f}'.format(eps) +  r', $\nu = $' + '{:.5f}'.format(nu) + r', $\alpha = $' + '{:.2f}'.format(alpha)

plt.figure()
plt.plot(processed['t'], processed['EN'] / k_force**2, color = 'orange', label = r'$Z_{\rm avg} / k_f^2 = \frac{1}{\pi k_f^2}\iint |\omega|^2 dA$')
plt.plot(processed['t'], processed['KE'], color = 'blue', label = r'$K_{\rm avg} = \frac{1}{\pi}\iint \frac{1}{2}|u|^2 dA$')
#plt.plot(processed['t'], (processed['EN_tavg'] / k_force**2) * np.ones(processed['t'].shape[0]), linestyle = 'dashed', color = 'red', label = r'$\langle Z_{\rm avg} \rangle / k_f^2$')
#plt.plot(processed['t'], processed['KE_tavg'] * np.ones(processed['t'].shape[0]), linestyle = 'dashed', color = 'black', label = r'$\langle K_{\rm avg} \rangle$') 
#plt.plot(processed['t'], processed['KE_growth_pred'], color ='purple', linestyle = "dotted", label = r'$\langle \partial_t K_{\rm avg} \rangle = (\epsilon / \pi) t$')
#plt.plot(processed['t'], processed['KE_tavg_expected'] * np.ones(processed['t'].shape[0]), linestyle = 'dashdot', color = 'purple', label = r'$\langle K_{\rm avg} \rangle = \frac{1}{2\alpha}\left(\frac{\epsilon}{\pi} - \nu Z_{\rm avg} \right)$')
#plt.plot(processed['t'], 8 * np.pi**2 * processed['KE_growth_pred'], color ='pink', linestyle = "dotted", label = r'$\langle \partial_t Z_{\rm avg} \rangle = (8 \epsilon \pi) t$')


plt.xlabel(r'$t$')
#plt.ylim(-0.5, 1.1*processed['KE_tavg_expected'])
#plt.ylim(-0.5, 7)
#plt.ylim(-0.2, 1)
plt.legend(loc = (1.05, 0.05), framealpha = 0.9)
plt.title(titlestr)
plt.tight_layout()
plt.savefig("KE.pdf")

print("eps/pi =", eps/np.pi, "nu * EN_avg = ", nu * processed['EN_tavg'])
print("Time-average value of eps/pi - nu * EN_avg", eps/np.pi - (nu * processed['EN_tavg']))
print("Time-average value of EN_avg / kf^2",  processed['EN_tavg'] / k_force**2)
print("Time-average value of EN_avg",  processed['EN_tavg'])
print("Time-average value of KE_avg",  processed['KE_tavg'])

#plt.figure()
#plt.plot(processed['t'], processed['Lzu'], color = 'blue')
#plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
#plt.xlabel('time')
#plt.ylabel('net angular momentum')
##plt.legend()
#plt.title("Time average of net angular momentum = {:.4}".format(np.mean(processed['Lzu'])))
#plt.tight_layout()
#plt.savefig("Lzu.pdf")

plt.figure()
plt.plot(processed['t'], processed['EN'], color = 'blue')
plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.xlabel('time')
plt.ylabel('Avg enstrophy')
#plt.legend()
plt.title('Enstrophy time series')
plt.tight_layout()
plt.savefig("EN.pdf")

#plt.figure()
#plt.plot(processed['t'], processed['W'], color = 'blue')
#plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
#plt.xlabel('time')
#plt.ylabel('net vorticity')
##plt.legend()
#plt.title("Time average of net vorticity = {:.4}".format(np.mean(processed['W'])))
#plt.tight_layout()
#plt.savefig("W.pdf")


#plt.figure()
##plt.plot(processed['t'], np.abs(processed['ENbdry']), label = 'ENbdry')
##plt.plot(processed['t'], np.abs(processed['PA']), label = 'PA')
##plt.plot(processed['t'], np.abs(processed['PAbdry1']), label = 'PAbdry1')
##plt.plot(processed['t'], np.abs(processed['PAbdry2']), label='PAbdry2')
#plt.plot(processed['t'], processed['ENbdry'], label = 'ENbdry')
#plt.plot(processed['t'], processed['PA'], label = 'PA')
#plt.plot(processed['t'], processed['PAbdry1'], label = 'PAbdry1')
#plt.plot(processed['t'], processed['PAbdry2'], label='PAbdry2')
#print(np.mean(processed['PA']), np.mean(processed['PAbdry1']), np.mean(processed['PAbdry2']))
#plt.xlabel('time')
##plt.yscale('log')
#plt.yscale('symlog')
#plt.legend()
#plt.tight_layout()
#plt.savefig("new.pdf")


seed_in = 10101
output_suffix = 'nu_{:.0e}'.format(nu)
output_suffix += '_gami_{:.1e}'.format(gammai)
output_suffix += '_gamf_{:.1e}'.format(gammaf)
output_suffix += '_dgam_{:.1e}'.format(dgamma)
output_suffix += '_kf_{:.1e}'.format(k_force)
output_suffix += '_Nphi_{:}'.format(Nphi)
output_suffix += '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_tau_mod_{:d}'.format(tau_mod)
output_suffix += '_seed_{:d}'.format(seed_in)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
processed_2 = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]

plt.figure()
plt.plot(processed['t'], processed['EN'] / k_force**2, color = 'orange', label = r'$Z_{\rm avg} / k_f^2 = \frac{1}{\pi k_f^2}\iint |\omega|^2 dA$')
plt.plot(processed['t'], processed['KE'], color = 'blue', label = r'$K_{\rm avg} = \frac{1}{\pi}\iint \frac{1}{2}|u|^2 dA$')
plt.plot(processed_2['t'], processed_2['EN'] / k_force**2, ls = 'dashed', color = 'red', label = r'$Z_{\rm avg} / k_f^2 = \frac{1}{\pi k_f^2}\iint |\omega|^2 dA$')
plt.plot(processed_2['t'], processed_2['KE'], ls = 'dashed', color = 'purple', label = r'$K_{\rm avg} = \frac{1}{\pi}\iint \frac{1}{2}|u|^2 dA$')
plt.xlabel(r'$t$')
plt.legend(loc = (1.05, 0.05), framealpha = 0.9)
plt.title(titlestr)
plt.tight_layout()
plt.savefig("KE_2.pdf")

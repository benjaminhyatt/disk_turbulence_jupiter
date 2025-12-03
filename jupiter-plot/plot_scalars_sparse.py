import numpy as np
import matplotlib.pyplot as plt
import h5py

Nphi, Nr = 1536, 768 #512, 256 #1024, 512#768, 384 #768, 384 #640, 320 #768, 384 #512, 256#1024, 512 
nu = 4e-5 #2e-4 #4e-5 #2e-4 #2e-4 #8e-5 #5e-4 #1e-3 #2e-4 #1e-4 #8e-5 #2e-4 #5e-5
gamma = 2500 #1920 #400 #675 #240 #675 #85 #675 #240 #2372 #1920#240 #675 #85 #1920 #240 #30 #0
k_force = 40 #20 #40 #20 #80 #20 #20 #10 #20 #70 #35 #20 #50

eps = 1
alpha = 3.3e-2 #1e-2

ring = 0

restart_evolved = False #False #True

restart_hyst = False #True
hystn = 7

#output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.0e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) + '_ring_0'
##output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

if restart_hyst:
    output_suffix += '_restart_hyst_{:d}'.format(hystn)

processed = np.load('../jupiter-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]


#print(processed['KE_tavg_expected'])
#processed['KE_tavg_expected'] = (1/(2*alpha)) * ((2/np.pi) - nu * processed['EN_tavg'])
#print(processed['KE_tavg_expected'])

if not restart_evolved:

    titlestr = r'$\gamma = $' + '{:.1f}'.format(gamma) + r', $k_f = $' +  '{:.1f}'.format(k_force) + r', $\epsilon = $' + '{:.1f}'.format(eps) +  r', $\nu = $' + '{:.5f}'.format(nu) + r', $\alpha = $' + '{:.2f}'.format(alpha)

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
else:
    plt.figure()
    plt.plot(processed['t'], processed['KE'], color = 'blue')
    plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
    plt.xlabel('time')
    plt.ylabel('Avg KE')
    #plt.legend()
    plt.title('Kinetic energy time series')
    plt.tight_layout()
    plt.savefig("KE.pdf")

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

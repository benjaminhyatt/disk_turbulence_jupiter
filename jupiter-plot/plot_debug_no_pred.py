import numpy as np
import matplotlib.pyplot as plt
import h5py
import dedalus.public as d3

#Nphi, Nr = 256, 128
Nphi, Nr = 512, 256
dealias = 3/2 
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
nu = 2e-4
alpha = 1e-2

processed = np.load('../jupiter-process/processed_debug_512_256_long_no_pred.npy', allow_pickle=True)[()]
times = processed['t']
plt.figure()
plt.plot(times, processed['lap_rate'], color = "cyan", label = r'Laplacian term from Integrate (M)')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
#plt.yscale('symlog', linthresh = 1e-5)
plt.savefig("lap_rate_debug_no_pred.pdf")

plt.figure()
plt.plot(times, processed['tau_rate'], color = "cyan", label = r'$n = N_r-1$ tau term from Integrate (M)')
plt.plot(times, processed['tau_rate_2'], color = "cyan", linestyle="dotted", label = r'$n = N_r-2$ tau term from Integrate (M)')
plt.legend()
#plt.ylim(-2e-4, 2e-4)
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("tau_rate_debug_no_pred.pdf")

plt.figure()
plt.plot(times, processed['tau_rate'] + processed['tau_rate_2'], color = "cyan", label = r'sum of tau terms from Integrate (M)')
plt.legend()
#plt.ylim(-2e-4, 2e-4)
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.yscale('symlog', linthresh=1e-16)
plt.tight_layout()
plt.savefig("tau_rate_combine_debug_no_pred.pdf")

plt.figure()
#plt.title(r'$-\sum_{n=0}^{N_r-1} \left((u\omega_z)_{0,n,-} + (u\omega_z)_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(times, processed['adv_rate'], color = "cyan", label = r'advection term from Integrate (M)')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("adv_rate_debug_no_pred.pdf")


plt.figure()
#plt.title(r'$-\sum_{n=0}^{N_r-1} \left(\tilde{F}_{0,n,-} - \tilde{F}_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(times, processed['for_rate'], color = "cyan", label = r'forcing term from Integrate (M)')
plt.xlabel(r'$t$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("for_rate_debug_no_pred.pdf")

# plot time-integrated analysis tasks, assuming that if the above check is true, that those time integrated calculations are accurate

plt.figure()
plt.plot(processed['t'], processed['W_lap'], color = 'blue', label = 'laplacian term')
plt.plot(processed['t'], processed['W_tau'], color = 'cyan', linewidth = 3, label = r'$n = N_r - 1$ tau term')
plt.plot(processed['t'], processed['W_tau_2'], color = 'purple', linewidth = 3, label = r'$n = N_r - 2$ tau term')
plt.plot(processed['t'], processed['W_adv'], color = 'red', label = 'advection term')
plt.plot(processed['t'], processed['W_for'], color = 'lime', linewidth = 3, label = 'forcing term')
plt.plot(processed['t'], processed['W_dam'], color = 'orange', linewidth = 3, label = r'damping term')
plt.plot(processed['t'], processed['W_sum'], color = 'gray', linewidth = 3, label = 'sum of above terms')
plt.plot(processed['t'], processed['W'], color = 'yellow', label = 'W')
plt.plot(processed['t'], np.zeros(processed['t'].shape[0]), linestyle = 'dashed', color = 'black')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$W$')
plt.tight_layout()
plt.savefig("W_debug_no_pred.pdf")

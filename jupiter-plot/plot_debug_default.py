import numpy as np
import matplotlib.pyplot as plt
import h5py
import dedalus.public as d3

Nphi, Nr = 256, 128
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

# load processed data dict
processed = np.load('../jupiter-process/processed_debug_default.npy', allow_pickle=True)[()]

# check scalars against d3 analysis tasks, to confirm we calculated the instantaneous rates correctly

plt.figure()
#plt.title(r'$-\nu\sum_{n=0}^{N_r-1} \left(\widetilde{\nabla^2 u}_{0,n,-} - \widetilde{\nabla^2 u}_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.title(r'$-\nu\sum_{n=0}^{N_r-1} \left(u_{0,n,-} - u_{0,n,+}\right) 2n(n+1)\sqrt{2n+1}(n^2 + 2n - 1)$')
#plt.plot(processed['t'], processed['lap_rate_pred_new'], color = "blue", label = r'From u coefficients')
plt.plot(processed['t'], processed['lap_rate_pred_old'], color = "red", linewidth = 3, label = r'$-\nu\partial_r \omega_0 (r=1)$')
plt.plot(processed['t'], processed['lap_rate'], color = "cyan", label = r'$\frac{1}{2\pi}$Integrate(-div(skew($\nu$lap(u))))')
plt.plot(processed['t'], processed['lap_rate_pred_new'], color = "blue", label = r'From u coefficients')
plt.plot(processed['t'], processed['lap_rate_pred'], color = "purple", label = r'From lap(u) coefficients')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.ylim(-1, 1)
plt.tight_layout()
#plt.yscale('symlog', linthresh = 1e-5)
plt.savefig("lap_rate_debug_default.pdf")

plt.figure()
plt.title(r'$\frac{2N_r}{\sqrt{2N_r-1}}(\widetilde{\tau u}_{0,-} - \widetilde{\tau u}_{0,+})$')
plt.plot(processed['t'], processed['tau_rate_pred'], color = "purple", linewidth = 3, label = 'From tau_u coefficients')
plt.plot(processed['t'], processed['tau_rate'], color = "red", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(-lift(tau_u))))')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("tau_rate_debug_default.pdf")

plt.figure()
plt.title(r'$-\sum_{n=0}^{N_r-1} \left((u\omega_z)_{0,n,-} + (u\omega_z)_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(processed['t'], processed['adv_rate_pred'], color = "red", linewidth = 3, label = 'From u wz coefficients')
plt.plot(processed['t'], processed['adv_rate'], color = "orange", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(-u$\cdot$ grad(u))))')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("adv_rate_debug_default.pdf")

plt.figure()
plt.title(r'$-\sum_{n=0}^{N_r-1} \left(\tilde{F}_{0,n,-} - \tilde{F}_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(processed['t'], processed['for_rate_pred'], color = "green", linewidth = 3, label = 'From F coefficients')
plt.plot(processed['t'], processed['for_rate'], color = "lime", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(F)))')
plt.xlabel(r'$t$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\partial_t W$')
plt.tight_layout()
plt.savefig("for_rate_debug_default.pdf")

# plot time-integrated analysis tasks, assuming that if the above check is true, that those time integrated calculations are accurate
plt.figure()
plt.title('W compared to time and volume integrated terms in Eqn 5')
plt.plot(processed['t'], processed['W_lap'], color = 'blue', label = 'laplacian term')
plt.plot(processed['t'], processed['W_tau'], color = 'cyan', linewidth = 3, label = 'tau term')
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
plt.savefig("W_debug_default.pdf")


# plot some profiles
plt.figure()
plt.plot(r[0, :], processed['w0_tavg'], label = 'time average')
plt.xlabel(r'$r$')
plt.ylabel(r'$\omega_0(r)$')
plt.tight_layout()
plt.savefig("w0_tavg_debug_default.pdf")

# plot some profiles
plt.figure()
plt.plot(r[0, :], processed['drw0_tavg'], label = 'time average')
plt.xlabel(r'$r$')
plt.ylabel(r'$\partial_r\omega_0(r)$')
plt.tight_layout()
plt.savefig("drw0_tavg_debug_default.pdf")

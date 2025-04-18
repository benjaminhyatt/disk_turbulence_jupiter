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

j0 = 0
j = -1

# load in analysis
#f = h5py.File('analysis_new_long/analysis_new_long_s1.h5')
f = h5py.File('analysis_new_longer/analysis_new_longer_s1.h5')
t = f['tasks/W'].dims[0]['sim_time'][j0:j]
print(t.shape)
tau_u = f['tasks/tau_u'][j0:j, :, :, 0] # tau_u_c
u_wz_c = f['tasks/u_wz_c'][j0:j, :, :, :]
F_c = f['tasks/F_c'][j0:j, :, :, :]

amp = -1
F_c *= amp

# scalars
W = f['tasks/W'][j0:j, 0, 0]
W *= 1/(2*np.pi)

drw0_1 = f['tasks/drw0_1'][j0:j, 0, 0]
F_1 = f['tasks/F_1'][j0:j, 0, 0]

# profiles
w0 = f['tasks/w0'][j0:j, 0, :]
drw0 = f['tasks/drw0'][j0:j, 0, :]
ur0 = f['tasks/ur0'][j0:j, 0, :]
ur0_drw0 = f['tasks/ur0_drw0'][j0:j, 0, :]

# process scalars
drw0_1 = nu*drw0_1
#tau_term = np.sqrt(2*(Nr-1))*(tau_u[:, 0, 1] - tau_u[:, 1, 1])
tau_term = (2*Nr / np.sqrt(2*Nr-1)) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
sig_Nr = - (2*Nr / np.sqrt(2*Nr-1)) * (np.sqrt(2*Nr-3) / (2*Nr-2))
tau_term_2 = sig_Nr * ((2*Nr-2) / np.sqrt(2*Nr-3)) * (tau_u[:, 0, 1] - tau_u[:, 1, 1])
aW_term = alpha*W

ur0_field = dist.Field(name='ur0', bases=radial_basis)
drw0_field = dist.Field(name='drw0', bases=radial_basis)
prods = np.zeros(t.shape[0])
for i in range(t.shape[0]):
    ur0_field.change_scales(1)
    drw0_field.change_scales(1)
    ur0_field['g'] = ur0[i, :]
    drw0_field['g'] = drw0[i, :]
    prod = ur0_field * drw0_field
    prods[i] = (1/(2*np.pi))*d3.integ(prod)['g'][0, 0]


ns = np.arange(0, Nr)
radial_integral = 2*(ns + 1)/np.sqrt(2*ns + 1)
u_wz_cos0_n_minus = u_wz_c[:, 0, 0, :]
u_wz_cos0_n_plus = u_wz_c[:, 1, 0, :]
F_sin0_n_minus = F_c[:, 0, 1, :]
F_sin0_n_plus = F_c[:, 1, 1, :]

adv_term = -np.sum((u_wz_cos0_n_minus + u_wz_cos0_n_plus)*radial_integral, axis = 1)
for_term = -np.sum((F_sin0_n_minus - F_sin0_n_plus)*radial_integral, axis = 1)


#f = h5py.File('analysis_aux_short/analysis_aux_short_s1.h5')
f = h5py.File('analysis_aux_longer/analysis_aux_longer_s1.h5')
W_wz_lap_rate = f['tasks/lap_rate'][j0:j, 0, 0]
W_wz_tau_rate = f['tasks/tau_rate'][j0:j, 0, 0]
W_wz_tau_rate_2 = f['tasks/tau_rate_2'][j0:j, 0, 0]
W_wz_adv_rate = f['tasks/adv_rate'][j0:j, 0, 0]
W_wz_for_rate = f['tasks/for_rate'][j0:j, 0, 0]
W_wz_dam_rate = f['tasks/dam_rate'][j0:j, 0, 0]
W_wz_lap_rate *= 1/(2*np.pi)
W_wz_tau_rate *= 1/(2*np.pi)
W_wz_tau_rate_2 *= 1/(2*np.pi)
W_wz_adv_rate *= 1/(2*np.pi)
W_wz_for_rate *= 1/(2*np.pi)
W_wz_dam_rate *= 1/(2*np.pi)

W_wz = f['tasks/wz'][j0:j, 0, 0]
W_wz_tau = f['tasks/W_tau'][j0:j, 0, 0]
W_wz_tau_2 = f['tasks/W_tau_2'][j0:j, 0, 0]
W_wz_lap = f['tasks/W_lap'][j0:j, 0, 0]
W_wz_for = f['tasks/W_for'][j0:j, 0, 0]
W_wz_adv = f['tasks/W_adv'][j0:j, 0, 0]
W_wz_dam = f['tasks/W_dam'][j0:j, 0, 0]
W_wz_lap *= 1/(2*np.pi)
W_wz_tau *= 1/(2*np.pi)
W_wz_tau_2 *= 1/(2*np.pi)
W_wz_adv *= 1/(2*np.pi)
W_wz_for *= 1/(2*np.pi)
W_wz_dam *= 1/(2*np.pi)

#print(W_wz_tau)
#print(W_wz_tau_2)

print(W_wz_tau_rate, W_wz_tau_rate_2)
print(W_wz_tau_rate + W_wz_tau_rate_2)

#print("Mean drw0_1 term:", np.mean(drw0_1), "Mean tau term:", np.mean(tau_term), "Mean F_1 term:", np.mean(-F_1), "Mean damping term:", np.mean(-aW_term), "Mean W:", np.mean(W))

# check scalars against d3 analysis tasks, to confirm we calculated the instantaneous rates correctly

# trust 
plt.figure()
plt.title(r'$\nu\partial\omega_0/\partial r (r=1)$')
plt.plot(t, drw0_1, color = "blue", linewidth = 3, label = r'Average(er$\cdot$grad(-div(skew($\nu$lap(u)))), phi)(r=1)')
plt.plot(t, W_wz_lap_rate, color = "cyan", label = r'$\frac{1}{2\pi}$Integrate(-div(skew($\nu$lap(u))))')
plt.tight_layout()
plt.legend()
plt.xlabel(r'$t$')
plt.savefig("eqn_5_lap_term.pdf")

# trust
plt.figure()
plt.title(r'$\frac{2N_r}{\sqrt{2N_r-1}}(\widetilde{\tau u}_{0,-} - \widetilde{\tau u}_{0,+})$')
plt.plot(t, tau_term, color = "purple", linewidth = 3, label = 'From tau_u coefficients')
plt.plot(t, W_wz_tau_rate, color = "red", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(-lift(tau_u))))')
plt.tight_layout()
plt.legend()
plt.xlabel(r'$t$')
plt.savefig("eqn_5_tau_term.pdf")

# new
plt.figure()
plt.title(r'$\sigma_{N_r}\frac{2N_r-2}{\sqrt{2N_r-3}}(\widetilde{\tau u}_{0,-} - \widetilde{\tau u}_{0,+})$')
plt.plot(t, tau_term_2, color = "purple", linewidth = 3, label = 'From tau_u coefficients')
plt.plot(t, W_wz_tau_rate_2, color = "red", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(-\sigma_{N_r} lift_2(tau_u))))')
plt.tight_layout()
plt.legend()
plt.xlabel(r'$t$')
plt.savefig("new_lift_term.pdf")

# not trusting yet
plt.figure()
#plt.title(r'$-\int_0^1 r u_{0,r} \frac{\partial\omega_0}{\partial r} dr$')
#plt.plot(t, -prods, color = "red", label = r'$\frac{1}{2\pi}$Integrate(Average(e_r$\cdot$ u, phi)Average(e_r$\cdot$grad$\omega_z$))')
plt.title(r'$-\sum_{n=0}^{N_r-1} \left((u\omega_z)_{0,n,-} + (u\omega_z)_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(t, adv_term, color = "red", linewidth = 3, label = 'From u wz coefficients')
plt.plot(t, W_wz_adv_rate, color = "orange", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(-u$\cdot$ grad(u))))')
plt.tight_layout()
plt.legend()
plt.xlabel(r'$t$')
plt.savefig("eqn_5_adv_term.pdf")

# not trusting yet
plt.figure()
#plt.title(r'$-F_{0,\phi}(r=1)$')
#plt.plot(t, -F_1, color = "green", label = 'Average(ephi$\cdot$F, phi)(r=1)')
plt.title(r'$-\sum_{n=0}^{N_r-1} \left(\tilde{F}_{0,n,-} - \tilde{F}_{0,n,+}\right) \frac{2(n+1)}{\sqrt{2n+1}}$')
plt.plot(t, for_term, color = "green", linewidth = 3, label = 'From F coefficients')
plt.plot(t, W_wz_for_rate, color = "lime", label = r'$\frac{1}{2\pi}$Integrate(-div(skew(F)))')
plt.tight_layout()
plt.xlabel(r'$t$')
plt.legend()
plt.xlabel(r'$t$')
plt.savefig("eqn_5_for_term.pdf")

# plot time-integrated analysis tasks, assuming that if the above check is true, that those time integrated calculations are accurate
max_abs_tau = np.max(np.abs(W_wz_tau_2))
mean_tau = np.mean(W_wz_tau_2)
print("max abs W_wz_tau =", max_abs_tau, "mean W_wz_tau", mean_tau) 
plt.figure()
#plt.title('W compared to time and volume integrated terms in Eqn 5\n' + f'max(abs(tau term)) = {max_abs_tau:.3}')
plt.title('W compared to time and volume integrated terms in Eqn 5')
plt.plot(t, W_wz_lap, color = 'blue', label = 'laplacian term')
plt.plot(t, W_wz_tau_2, color = 'cyan', linewidth = 3, label = 'tau term (including modification)')
plt.plot(t, W_wz_adv, color = 'red', label = 'advection term')
plt.plot(t, W_wz_for, color = 'green', linewidth = 3, label = 'forcing term')
plt.plot(t, W_wz_dam, color = 'orange', linewidth = 3, label = r'damping term')
plt.plot(t, W_wz_lap + W_wz_tau_2 + W_wz_adv + W_wz_for + W_wz_dam, color = 'gray', linewidth = 3, label = 'sum of terms (including modification)')
plt.plot(t, W, color = 'yellow', label = 'W (including modification)')
plt.plot(t, np.zeros(t.shape[0]), linestyle = 'dashed', color = 'black')
plt.legend()
plt.xlabel(r'$t$')
plt.tight_layout()
plt.savefig("W_comparison_mod.pdf")


# plot time-integrated analysis tasks, assuming that if the above check is true, that those time integrated calculations are accurate
max_abs_tau = np.max(np.abs(W_wz_tau_2))
mean_tau = np.mean(W_wz_tau_2)
print("max abs W_wz_tau =", max_abs_tau, "mean W_wz_tau", mean_tau)
plt.figure()
#plt.title('W compared to time and volume integrated terms in Eqn 5\n' + f'max(abs(tau term)) = {max_abs_tau:.3}')
plt.title('W compared to time and volume integrated terms in Eqn 5')
plt.plot(t, W_wz_lap, color = 'blue', label = 'laplacian term')
plt.plot(t, W_wz_tau_2, color = 'cyan', linewidth = 3, label = 'tau term (including modification)')
plt.plot(t, W_wz_adv, color = 'red', label = 'advection term')
plt.plot(t, W_wz_for, color = 'green', linewidth = 3, label = 'forcing term')
plt.plot(t, W_wz_dam, color = 'orange', linewidth = 3, label = r'damping term')
plt.plot(t, W_wz_lap + W_wz_tau_2 + W_wz_adv + W_wz_for + W_wz_dam, color = 'gray', linewidth = 3, label = 'sum of terms (including modification)')
plt.plot(t, W, color = 'yellow', label = 'W (including modification)')
plt.plot(t, np.zeros(t.shape[0]), linestyle = 'dashed', color = 'black')
plt.legend()
plt.ylim(-1.2*np.max(np.abs(W)), 1.2*np.max(np.abs(W)))
plt.xlabel(r'$t$')
plt.tight_layout()
plt.savefig("W_comparison_mod_zoom.pdf")

# plot snapshots of omega profiles
w0_tavg = np.mean(w0, axis = 0)
plt.figure()
plt.xlabel(r'$r$')
plt.ylabel(r'$\omega_0(r)$')
plt.title('radial profiles')
#plt.plot(r[0,:], w0[100, :], label = r'$t=20$')
#plt.plot(r[0,:], w0[200, :], label = r'$t=40$')
#plt.plot(r[0,:], w0[300, :], label = r'$t=60$')
#plt.plot(r[0,:], w0[400, :], label = r'$t=80$')
plt.plot(r[0, :], w0_tavg, label = r'time average')
plt.legend()
plt.tight_layout()
plt.savefig("w0_tavg.pdf")
drw0_tavg = np.mean(drw0, axis = 0)
plt.figure()
plt.xlabel(r'$r$')
plt.ylabel(r'$\partial\omega_0(r)/\partial r$')
plt.title('radial profiles')
#plt.plot(r[0,:], drw0[100, :], label = r'$t=20$')
#plt.plot(r[0,:], drw0[200, :], label = r'$t=40$')
#plt.plot(r[0,:], drw0[300, :], label = r'$t=60$')
#plt.plot(r[0,:], drw0[400, :], label = r'$t=80$')
plt.plot(r[0, :], drw0_tavg, label = r'time average')
plt.legend()
plt.tight_layout()
plt.savefig("drw0_tavg.pdf")

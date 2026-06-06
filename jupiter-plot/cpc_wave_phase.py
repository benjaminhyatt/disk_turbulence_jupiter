"""
Phase relationship diagnostic between CPC tracking angle and dominant Rossby
wave projection amplitude, for gamma = 675.

Approach: treat both signals as phasors and measure their phase difference
directly, without unwrapping or co-rotating frame computation.

Usage:
    python cpc_wave_phase.py

Outputs:
    - cpc_wave_phase.png : figure with 4 panels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks

# ── Load data ────────────────────────────────────────────────────────────────

tracking_file = (
    "../jupiter-process/"
    "processed_tracking_nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_"
    "eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_1_tau_mod_1_"
    "seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

projection_file = (
    "../jupiter-process/"
    "processed_rossby_projection_mpm_sort_im_inc_m_1_inviscid_0_u0bg_0_"
    "nu_2em04_gam_6d8ep02_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_"
    "alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_10001_"
    "safety_1d0em01_timestepper_SBDF2_bc_sf.npy"
)

tracking  = np.load(tracking_file,  allow_pickle=True)[()]
projected = np.load(projection_file, allow_pickle=True)[()]

# ── Extract signals ───────────────────────────────────────────────────────────

t       = tracking['tw']
phi_raw = np.array(tracking['phi_locs'], dtype=float)   # raw angle in [0, 2pi)
r_cpc   = np.array(tracking['r_locs'],   dtype=float)
A_dom   = projected['projl2'][0, :]                     # dominant mode, sorted by amplitude

t_proj = projected['tw']
if not np.allclose(t, t_proj, rtol=1e-6):
    print("WARNING: time axes do not match exactly")
else:
    print(f"Time axes agree: {len(t)} points, t in [{t[0]:.3f}, {t[-1]:.3f}]")

dt = np.mean(np.diff(t))
print(f"dt = {dt:.5f},  T_window = {t[-1]-t[0]:.3f}")

# ── Orbital frequency from consecutive phasor differences ────────────────────
# This is unwrap-free: each step dphi is in (-pi, pi], safe as long as the
# CPC moves less than half a revolution per timestep (easily satisfied here).

z_cpc     = np.exp(- 1j * phi_raw)
dphi      = np.angle(z_cpc[1:] * np.conj(z_cpc[:-1]))
Omega_CPC = np.mean(dphi) / dt
T_CPC     = 2 * np.pi / np.abs(Omega_CPC)

print(f"\nMean CPC orbital angular velocity: Omega_CPC = {Omega_CPC:.6f} rad/time")
print(f"Mean CPC orbital period:           T_CPC     = {T_CPC:.4f} time units")
print(f"Number of orbits in window:        {(t[-1]-t[0])/T_CPC:.2f}")

# ── Outlier detection ─────────────────────────────────────────────────────────
# Flag timesteps where the instantaneous angular speed deviates strongly from
# the mean, or where r_cpc drops far below its mean — these are tracking glitches.

omega_inst  = dphi / dt                            # shape (N-1,)
omega_mean  = np.abs(Omega_CPC)
omega_std   = np.std(omega_inst)
r_mean      = np.mean(r_cpc)

# glitch mask on the (N-1) interval grid, then broadcast to N points
glitch_omega = np.abs(omega_inst - Omega_CPC) > 5 * omega_std
glitch_r     = r_cpc < 0.5 * r_mean

# expand interval mask to point mask
glitch_mask       = np.zeros(len(t), dtype=bool)
glitch_mask[:-1] |= glitch_omega
glitch_mask[1:]  |= glitch_omega
glitch_mask      |= glitch_r

n_glitch = np.sum(glitch_mask)
print(f"\nGlitch points detected: {n_glitch} / {len(t)}  ({100*n_glitch/len(t):.1f}%)")

# ── Analytic signal of A_dom via Hilbert transform ───────────────────────────
# The Hilbert transform gives us the analytic signal z_A = A_dom + i*H[A_dom],
# whose instantaneous phase tracks the oscillation of A_dom continuously.

z_A      = hilbert(A_dom)                          # complex analytic signal
phase_A  = np.angle(z_A)                           # instantaneous phase of A_dom

# ── Phase difference CPC vs wave ─────────────────────────────────────────────
# z_cpc = e^{i phi_CPC} already has unit magnitude.
# Normalise z_A to unit magnitude before taking the angle difference.

z_A_unit  = z_A / np.abs(z_A)
delta_phi = np.angle(z_cpc * np.conj(z_A_unit))   # phase of CPC minus phase of wave

# Statistics excluding glitch points
delta_phi_clean = delta_phi[~glitch_mask]
mean_delta      = np.mean(delta_phi_clean)
std_delta       = np.std(delta_phi_clean)
median_delta    = np.median(delta_phi_clean)

print(f"\nPhase offset  phi_CPC - phi_wave  (glitches excluded):")
print(f"  mean   = {mean_delta:.4f} rad  ({np.degrees(mean_delta):.2f} deg)")
print(f"  median = {median_delta:.4f} rad  ({np.degrees(median_delta):.2f} deg)")
print(f"  std    = {std_delta:.4f} rad  ({np.degrees(std_delta):.2f} deg)")
print(f"  (0 = CPC at wave crest, pi/2 = CPC at leading flank, pi = CPC at trough)")

# ── Frequency from A_dom FFT ──────────────────────────────────────────────────

freqs_A  = np.fft.rfftfreq(len(t), d=dt) * 2 * np.pi
fft_A    = np.fft.rfft(A_dom - np.mean(A_dom))
power_A  = np.abs(fft_A)**2
peak_idx = np.argmax(power_A)
omega_A  = freqs_A[peak_idx]

print(f"\nDominant frequency in A_dom:  omega_A = {omega_A:.4f} rad/time")
print(f"  cf. |Omega_CPC|           = {np.abs(Omega_CPC):.4f} rad/time")
print(f"  fractional difference     = {abs(omega_A - np.abs(Omega_CPC))/np.abs(Omega_CPC)*100:.3f}%")

# ── r_CPC statistics ──────────────────────────────────────────────────────────

r_clean = r_cpc[~glitch_mask]
print(f"\nr_CPC (glitches excluded):")
print(f"  mean = {np.mean(r_clean):.5f}")
print(f"  std  = {np.std(r_clean):.5f}")
print(f"  min  = {np.min(r_clean):.5f},  max = {np.max(r_clean):.5f}")

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

# Panel 1: raw signals overlaid, both normalised to [-1, 1] by their max abs
ax = axes[0]
phi_plot = phi_raw / np.pi        # in units of pi, so range is [0, 2]
phi_plot -= np.mean(phi_plot)     # centre for overlay
phi_plot /= np.max(np.abs(phi_plot))
A_plot    = A_dom / np.max(np.abs(A_dom))

ax.plot(t, A_plot,   color='C0', lw=1.2, label=r'$A_\mathrm{dom}$ (norm.)')
ax.plot(t, phi_plot, color='C1', lw=1.2, label=r'$\phi_\mathrm{CPC}$ (norm., centred)',
        alpha=0.8)
# mark glitches
if n_glitch > 0:
    ax.scatter(t[glitch_mask], phi_plot[glitch_mask], color='red', s=10, zorder=5,
               label='detected glitches')
ax.set_xlabel('time')
ax.set_ylabel('normalised amplitude')
ax.set_title('Raw signals: dominant mode amplitude vs. CPC angle')
ax.legend(fontsize=9)

# Panel 2: r_CPC with glitches marked
ax = axes[1]
ax.plot(t, r_cpc, color='C2', lw=1.0, label=r'$r_\mathrm{CPC}$')
ax.axhline(r_mean,       color='C2', ls='--', lw=0.8,
           label=f'mean = {r_mean:.4f}')
ax.axhline(0.5*r_mean,   color='red', ls=':', lw=0.8,
           label=f'glitch threshold = {0.5*r_mean:.4f}')
if n_glitch > 0:
    ax.scatter(t[glitch_mask], r_cpc[glitch_mask], color='red', s=10, zorder=5)
ax.set_xlabel('time')
ax.set_ylabel(r'$r_\mathrm{CPC}$')
ax.set_title('CPC radial position')
ax.legend(fontsize=9)

# Panel 3: instantaneous phase difference over time
ax = axes[2]
ax.plot(t[~glitch_mask], np.degrees(delta_phi[~glitch_mask]),
        color='C3', lw=0.8, label='phase offset (clean)')
if n_glitch > 0:
    ax.scatter(t[glitch_mask], np.degrees(delta_phi[glitch_mask]),
               color='red', s=10, zorder=5, label='glitch points')
ax.axhline(np.degrees(mean_delta),   color='k',    ls='--', lw=1.0,
           label=f'mean = {np.degrees(mean_delta):.1f}°')
ax.axhline(np.degrees(mean_delta) + np.degrees(std_delta),
           color='gray', ls=':', lw=0.8)
ax.axhline(np.degrees(mean_delta) - np.degrees(std_delta),
           color='gray', ls=':', lw=0.8, label=f'±1σ = {np.degrees(std_delta):.1f}°')
ax.set_xlabel('time')
ax.set_ylabel('phase offset (degrees)')
ax.set_title(r'Instantaneous phase offset: $\phi_\mathrm{CPC} - \phi_\mathrm{wave}$')
ax.legend(fontsize=9)

# Panel 4: power spectrum of A_dom with Omega_CPC marked
ax = axes[3]
mask_f = freqs_A <= 3 * np.abs(Omega_CPC)
ax.plot(freqs_A[mask_f], power_A[mask_f] / np.max(power_A),
        color='C0', lw=1.2, label=r'$A_\mathrm{dom}$ power spectrum')
ax.axvline(np.abs(Omega_CPC), color='C1', ls='--', lw=1.0,
           label=f'$|\\Omega_{{\\rm CPC}}|$ = {np.abs(Omega_CPC):.4f}')
ax.axvline(omega_A,           color='C0', ls=':',  lw=1.0,
           label=f'$\\omega_A$ = {omega_A:.4f}')
ax.set_xlabel(r'$\omega$ (rad / time)')
ax.set_ylabel('normalised power')
ax.set_title(r'Power spectrum of $A_\mathrm{dom}$')
ax.legend(fontsize=9)

fig.suptitle(
    r'$\gamma = 675$ — CPC phase vs. dominant $m=1$ Rossby mode',
    fontsize=12
)

outname = 'cpc_wave_phase.png'
fig.savefig(outname, dpi=150)
print(f"\nFigure saved to {outname}")
plt.close(fig)

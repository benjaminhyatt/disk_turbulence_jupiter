"""
Cross-correlation diagnostic between CPC tracking angle and dominant Rossby wave
projection amplitude, for a given gamma simulation.

Revised approach: use exp(i*phi_CPC) rather than unwrapping phi, to avoid
accumulation of unwrapping errors over many orbital periods.

Usage:
    python cpc_wave_crosscorr_v2.py

Outputs:
    - cpc_wave_crosscorr_v2.png : figure with 4 panels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
phi_raw = np.array(tracking['phi_locs'])   # CPC azimuthal angle (wrapped, in [-pi, pi] or [0, 2pi])
r_cpc   = np.array(tracking['r_locs'])

A_dom   = projected['projl2'][0, :]   # dominant mode amplitude, sorted by amplitude

t_proj = projected['tw']
if not np.allclose(t, t_proj, rtol=1e-6):
    print("WARNING: time axes do not match exactly")
else:
    print(f"Time axes agree: {len(t)} points, t in [{t[0]:.3f}, {t[-1]:.3f}]")

dt = np.mean(np.diff(t))
print(f"dt = {dt:.5f},  T_window = {t[-1]-t[0]:.3f}")

# ── Estimate orbital frequency from phi directly ──────────────────────────────
# Use the complex exponential e^{i*phi} — its instantaneous frequency is
# d(phi)/dt, which we get cleanly from the angle of consecutive ratios.

z_phi       = np.exp(1j * phi_raw)              # unit-magnitude phasor
# instantaneous angular step per dt (unwrap-free, stays in (-pi, pi])
dphi        = np.angle(z_phi[1:] * np.conj(z_phi[:-1]))
Omega_inst  = dphi / dt                          # instantaneous angular velocity
Omega_CPC   = np.mean(Omega_inst)               # mean orbital angular velocity
T_CPC       = 2 * np.pi / np.abs(Omega_CPC)

print(f"\nMean CPC orbital angular velocity: Omega_CPC = {Omega_CPC:.6f} rad/time")
print(f"Mean CPC orbital period:           T_CPC     = {T_CPC:.4f} time units")
print(f"Number of orbits in window:        {(t[-1]-t[0])/T_CPC:.2f}")

# ── Build co-rotating phasor ──────────────────────────────────────────────────
# Remove the mean orbital drift by multiplying by e^{-i Omega_CPC t}.
# This leaves only the slow residual fluctuation (libration, noise, etc.)

z_corot  = z_phi * np.exp(-1j * Omega_CPC * t)   # complex residual in co-rotating frame
phi_resid = np.angle(z_corot)                      # real residual angle in (-pi, pi]

# ── Detrend A_dom ─────────────────────────────────────────────────────────────

A_detrended = A_dom - np.mean(A_dom)

# ── Cross-correlation: phi_resid vs A_dom ────────────────────────────────────

phi_norm = phi_resid   / (np.std(phi_resid)   + 1e-14)
A_norm   = A_detrended / (np.std(A_detrended) + 1e-14)

xcorr    = signal.correlate(phi_norm, A_norm, mode='full') / len(t)
lags     = signal.correlation_lags(len(t), len(t), mode='full')
lag_time = lags * dt

# Focus on lags within +/- 2 orbital periods
lag_mask  = np.abs(lag_time) <= 2 * T_CPC
peak_idx  = np.argmax(np.abs(xcorr[lag_mask]))
peak_lag  = lag_time[lag_mask][peak_idx]
peak_corr = xcorr[lag_mask][peak_idx]

print(f"\nCross-correlation peak (within ±2 T_CPC):")
print(f"  value = {peak_corr:.4f}  at lag = {peak_lag:.4f} time units")
print(f"  lag as fraction of T_CPC: {peak_lag/T_CPC:.4f}")

# ── Power spectra ─────────────────────────────────────────────────────────────

freqs   = np.fft.rfftfreq(len(t), d=dt) * 2 * np.pi   # angular frequencies rad/time
fft_phi = np.fft.rfft(phi_resid)
fft_A   = np.fft.rfft(A_detrended)

# also FFT of instantaneous omega to show orbital frequency content
freqs_o  = np.fft.rfftfreq(len(Omega_inst), d=dt) * 2 * np.pi
fft_O    = np.fft.rfft(Omega_inst - Omega_CPC)

# dominant peaks
def top_peaks(freqs, power, n=5):
    from scipy.signal import find_peaks
    pks, _ = find_peaks(power, height=0.05*np.max(power))
    if len(pks) == 0:
        return []
    order = np.argsort(power[pks])[::-1]
    return [(freqs[pks[i]], power[pks[i]]) for i in order[:n]]

print("\nTop spectral peaks in phi_resid:")
for f, p in top_peaks(freqs, np.abs(fft_phi)**2):
    print(f"  omega = {f:.4f}  (power = {p:.3e})")

print("\nTop spectral peaks in A_dom:")
for f, p in top_peaks(freqs, np.abs(fft_A)**2):
    print(f"  omega = {f:.4f}  (power = {p:.3e})")

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

# Panel 1: co-rotating residual phi vs A_dom (both normalised)
ax = axes[0]
ax.plot(t, phi_norm, label=r'$\phi_\mathrm{CPC}$ co-rotating residual (norm.)', lw=1.2)
ax.plot(t, A_norm,   label=r'$A_\mathrm{dom}$ detrended (norm.)',               lw=1.2, alpha=0.8)
ax.set_xlabel('time')
ax.set_ylabel('normalised amplitude')
ax.set_title('Co-rotating CPC angle residual vs. dominant mode amplitude')
ax.legend(fontsize=9)

# Panel 2: r_CPC(t) and instantaneous orbital rate
ax = axes[1]
ax2b = ax.twinx()
ax.plot(t, r_cpc,         color='C2', lw=1.2, label=r'$r_\mathrm{CPC}$')
ax2b.plot(t[:-1] + dt/2, Omega_inst, color='C4', lw=0.8, alpha=0.6,
          label=r'$\dot\phi_\mathrm{CPC}$')
ax.axhline(np.mean(r_cpc), color='C2', ls='--', lw=0.8,
           label=f'$\\langle r \\rangle$ = {np.mean(r_cpc):.4f}')
ax.set_xlabel('time')
ax.set_ylabel(r'$r_\mathrm{CPC}$', color='C2')
ax2b.set_ylabel(r'$\dot\phi$ (rad/time)', color='C4')
ax.set_title(r'CPC radial position and instantaneous angular velocity')
ax.legend(loc='upper left', fontsize=9)
ax2b.legend(loc='upper right', fontsize=9)

# Panel 3: cross-correlation vs lag (zoomed to ±2 T_CPC)
ax = axes[2]
ax.plot(lag_time[lag_mask], xcorr[lag_mask], color='C3', lw=1.2)
ax.axvline(peak_lag, color='k', ls='--', lw=0.8,
           label=f'peak lag = {peak_lag:.4f} ({peak_lag/T_CPC:.3f} $T_{{\\rm CPC}}$)')
ax.axhline(0, color='gray', lw=0.5)
for k in [-1, 0, 1]:
    ax.axvline(k * T_CPC, color='gray', ls=':', lw=0.6)
ax.set_xlabel('lag (time units)')
ax.set_ylabel('cross-correlation')
ax.set_title(r'Cross-correlation (zoomed to $\pm 2\, T_\mathrm{CPC}$)')
ax.legend(fontsize=9)

# Panel 4: power spectra (zoomed to relevant frequency range)
ax = axes[3]
omega_max_plot = 5 * np.abs(Omega_CPC)
mask_f = freqs <= omega_max_plot
ax.plot(freqs[mask_f], np.abs(fft_phi[mask_f])**2 / np.max(np.abs(fft_phi)**2),
        label=r'$\phi_\mathrm{CPC}$ residual', lw=1.2)
ax.plot(freqs[mask_f], np.abs(fft_A[mask_f])**2   / np.max(np.abs(fft_A)**2),
        label=r'$A_\mathrm{dom}$',              lw=1.2, alpha=0.8)
ax.axvline(np.abs(Omega_CPC), color='k',  ls='--', lw=0.8,
           label=f'$|\\Omega_{{\\rm CPC}}|$ = {np.abs(Omega_CPC):.3f}')
ax.set_xlabel(r'$\omega$ (rad / time)')
ax.set_ylabel('normalised power')
ax.set_title('Power spectra of residual signals')
ax.legend(fontsize=9)

fig.suptitle(
    r'$\gamma = 675$ — CPC tracking vs. dominant $m=1$ Rossby mode (v2)',
    fontsize=12
)

outname = 'cpc_wave_crosscorr_v2.png'
fig.savefig(outname, dpi=150)
print(f"\nFigure saved to {outname}")
plt.close(fig)

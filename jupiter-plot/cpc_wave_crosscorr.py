"""
Cross-correlation diagnostic between CPC tracking angle and dominant Rossby wave
projection amplitude, for a given gamma simulation.

Usage:
    python cpc_wave_crosscorr.py

Outputs:
    - cpc_wave_crosscorr.png : figure with 3 panels
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

t       = tracking['tw']           # shared time axis
phi_raw = tracking['phi_locs']    # CPC azimuthal angle, possibly wrapped
r_cpc   = tracking['r_locs']      # CPC radial position

A_dom   = projected['projl2'][0, :]   # dominant mode amplitude (sorted by amplitude)

# Sanity check: confirm time axes agree
t_proj = projected['tw']
if not np.allclose(t, t_proj, rtol=1e-6):
    print("WARNING: time axes do not match exactly — check alignment")
else:
    print(f"Time axes agree: {len(t)} points, t in [{t[0]:.3f}, {t[-1]:.3f}]")

dt = np.mean(np.diff(t))
print(f"dt = {dt:.4f},  T_window = {t[-1]-t[0]:.3f}")

# ── Unwrap and detrend phi ────────────────────────────────────────────────────

phi_unwrapped = np.unwrap(phi_raw)

# Linear fit to unwrap to get mean orbital rate
p         = np.polyfit(t, phi_unwrapped, 1)
Omega_CPC = p[0]
phi_trend = np.polyval(p, t)
phi_resid = phi_unwrapped - phi_trend   # fluctuations around mean orbit

print(f"Mean CPC orbital angular velocity: Omega_CPC = {Omega_CPC:.6f} rad/time")
print(f"Mean CPC orbital period: T_CPC = {2*np.pi/np.abs(Omega_CPC):.3f} time units")

# ── Detrend A_dom ─────────────────────────────────────────────────────────────

A_detrended = A_dom - np.mean(A_dom)

# ── Cross-correlation ─────────────────────────────────────────────────────────

# Normalise both signals to unit variance for interpretable correlation coefficient
phi_norm = phi_resid  / (np.std(phi_resid)  + 1e-14)
A_norm   = A_detrended / (np.std(A_detrended) + 1e-14)

# Full cross-correlation
xcorr    = signal.correlate(phi_norm, A_norm, mode='full') / len(t)
lags     = signal.correlation_lags(len(t), len(t), mode='full')
lag_time = lags * dt

# Find peak
peak_idx  = np.argmax(np.abs(xcorr))
peak_lag  = lag_time[peak_idx]
peak_corr = xcorr[peak_idx]

print(f"\nCross-correlation peak: {peak_corr:.4f} at lag = {peak_lag:.4f} time units")
print(f"  (positive lag means phi_resid leads A_dom)")

# ── FFT of both signals ───────────────────────────────────────────────────────

freqs   = np.fft.rfftfreq(len(t), d=dt) * 2 * np.pi   # angular frequencies
fft_phi = np.fft.rfft(phi_resid)
fft_A   = np.fft.rfft(A_detrended)

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(10, 13), constrained_layout=True)

# Panel 1: raw signals overlaid (normalised)
ax = axes[0]
ax.plot(t, phi_norm, label=r'$\phi_\mathrm{CPC}$ residual (normalised)', lw=1.2)
ax.plot(t, A_norm,   label=r'$A_\mathrm{dom}$ detrended (normalised)',   lw=1.2, alpha=0.8)
ax.set_xlabel('time')
ax.set_ylabel('normalised amplitude')
ax.set_title('CPC angle residual vs. dominant mode amplitude')
ax.legend(fontsize=9)

# Panel 2: r_CPC(t)
ax = axes[1]
ax.plot(t, r_cpc, color='C2', lw=1.2)
ax.set_xlabel('time')
ax.set_ylabel(r'$r_\mathrm{CPC}$')
ax.set_title('CPC radial position')
ax.axhline(np.mean(r_cpc), color='k', ls='--', lw=0.8, label=f'mean = {np.mean(r_cpc):.4f}')
ax.legend(fontsize=9)

# Panel 3: cross-correlation vs lag
ax = axes[2]
# only show +/- 2 orbital periods
T_orb    = 2 * np.pi / np.abs(Omega_CPC)
lag_mask = np.abs(lag_time) <= 2 * T_orb
ax.plot(lag_time[lag_mask], xcorr[lag_mask], color='C3', lw=1.2)
ax.axvline(peak_lag, color='k', ls='--', lw=0.8,
           label=f'peak lag = {peak_lag:.3f}')
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('lag (time units)')
ax.set_ylabel('cross-correlation')
ax.set_title(r'Cross-correlation: $\phi_\mathrm{CPC}$ residual $\star$ $A_\mathrm{dom}$')
ax.legend(fontsize=9)

# Panel 4: power spectra of both signals
ax = axes[3]
ax.plot(freqs, np.abs(fft_phi)**2 / np.max(np.abs(fft_phi)**2),
        label=r'$\phi_\mathrm{CPC}$ residual', lw=1.2)
ax.plot(freqs, np.abs(fft_A)**2   / np.max(np.abs(fft_A)**2),
        label=r'$A_\mathrm{dom}$',              lw=1.2, alpha=0.8)
# mark Omega_CPC
ax.axvline(np.abs(Omega_CPC), color='k', ls='--', lw=0.8,
           label=f'$\\Omega_\\mathrm{{CPC}}$ = {np.abs(Omega_CPC):.4f}')
ax.set_xlabel(r'$\omega$ (rad / time)')
ax.set_ylabel('normalised power')
ax.set_title('Power spectra')
ax.set_xlim([0, 10 * np.abs(Omega_CPC)])   # focus on relevant frequency range
ax.legend(fontsize=9)

fig.suptitle(
    r'$\gamma = 675$ — CPC tracking vs. dominant $m=1$ Rossby mode',
    fontsize=12
)

outname = 'cpc_wave_crosscorr.png'
fig.savefig(outname, dpi=150)
print(f"\nFigure saved to {outname}")
plt.close(fig)

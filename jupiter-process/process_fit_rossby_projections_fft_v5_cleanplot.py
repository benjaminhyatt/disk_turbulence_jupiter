"""
Fit oscillation frequencies of Rossby mode projection amplitudes using FFT,
loading from pre-processed projection output.

Peak selection logic:
  1. Find top N_peaks peaks by power above min_freq (full spectrum saved).
  2. Among those, select the peak whose frequency is closest to the EVP
     eigenvalue real part for that mode, subject to:
       - frequency must be within evp_frac_window of the EVP prediction
       - power must be at least min_power_frac of the strongest peak
     If no peak passes these criteria, fall back to the strongest peak.
  3. After selection, flag any retained peak whose frequency falls within
     harmonic_frac of n * omega_selected for n = 2, 3 as a likely harmonic.

Amplitude estimation:
  - Primary: extrema-based estimate -- mean of absolute values of local
    extrema of projdot. Robust to beat patterns and harmonic contamination.
  - Secondary: single-sinusoid least-squares fit amplitude at omega_selected.
    Retained for reference but biased low when amplitude modulation is present.

Panel 4 shows the normalized fit residual (projdot - fit) / amp_extrema,
making the beat pattern and fit quality directly visible.

Usage:
    process_fit_rossby_projections_fft.py <processed_file> [options]

Options:
    --output=<str>              output file prefix [default: processed_rossby_projection_fft]
    --N_peaks=<int>             number of peaks to retain per mode [default: 5]
    --min_freq=<float>          minimum frequency to consider (rad/time) [default: 0.5]
    --centroid_bins=<int>       half-width in bins for centroid window [default: 4]
    --gauss_bins=<int>          half-width in bins for Gaussian fit window [default: 6]
    --evp_frac_window=<float>   fractional window around EVP eigenvalue for peak selection [default: 0.35]
    --min_power_frac=<float>    minimum power fraction relative to strongest peak [default: 0.05]
    --harmonic_frac=<float>     fractional window for harmonic flagging around n*omega [default: 0.05]
    --plot=<bool>               save summary figure per mode [default: True]
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from docopt import docopt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

### read in options ###
args = docopt(__doc__)
logger.info("args read in")
print(args)

processed_file_str  = args['<processed_file>']
output              = args['--output']
N_peaks             = int(args['--N_peaks'])
min_freq            = float(args['--min_freq'])
centroid_bins       = int(args['--centroid_bins'])
gauss_bins          = int(args['--gauss_bins'])
evp_frac_window     = float(args['--evp_frac_window'])
min_power_frac      = float(args['--min_power_frac'])
harmonic_frac       = float(args['--harmonic_frac'])
plot                = eval(args['--plot'])

output_suffix = processed_file_str.split('processed_rossby_projection_')[1].split('.')[0].split('/')[0]
output_prefix = output

### helper functions ###

def real_fft_spectrum(y, dt, window=None):
    N = len(y)
    if window is not None:
        y = y * window * N / np.sum(window)
    fft_y = np.fft.rfft(y) / N
    freqs  = np.fft.rfftfreq(N, d=dt) * 2 * np.pi
    return freqs, fft_y

def quadratic_interpolate_peak(freqs, power, peak_idx):
    k = peak_idx
    if k == 0 or k == len(freqs) - 1:
        return freqs[k]
    d_omega = freqs[1] - freqs[0]
    Pm, Pk, Pp = power[k-1], power[k], power[k+1]
    denom = 2 * (2*Pk - Pm - Pp)
    if np.abs(denom) < 1e-30:
        return freqs[k]
    return freqs[k] + (Pp - Pm) / denom * d_omega

def centroid_frequency(freqs, power, peak_idx, half_width):
    N  = len(freqs)
    lo = max(0, peak_idx - half_width)
    hi = min(N - 1, peak_idx + half_width)
    w  = power[lo:hi+1]
    f  = freqs[lo:hi+1]
    denom = np.sum(w)
    if denom < 1e-30:
        return freqs[peak_idx]
    return np.sum(f * w) / denom

def gaussian_log_fit(freqs, power, peak_idx, half_width):
    N  = len(freqs)
    lo = max(0, peak_idx - half_width)
    hi = min(N - 1, peak_idx + half_width)
    f  = freqs[lo:hi+1]
    p  = power[lo:hi+1]
    p  = np.where(p > 1e-30 * np.max(p), p, 1e-30 * np.max(p))
    log_p = np.log(p)
    def log_gauss(x, log_A, mu, sigma):
        return log_A - 0.5 * ((x - mu) / sigma)**2
    try:
        p0     = [log_p[np.argmax(log_p)], freqs[peak_idx], (freqs[hi] - freqs[lo]) / 4]
        bounds = ([-np.inf, freqs[lo], 1e-6], [np.inf, freqs[hi], freqs[hi]-freqs[lo]])
        popt, _ = curve_fit(log_gauss, f, log_p, p0=p0, bounds=bounds, maxfev=2000)
        return popt[1]
    except Exception:
        return None

def find_top_peaks(freqs, power, freqs_hann, power_hann,
                   N_peaks, min_freq, centroid_bins, gauss_bins):
    mask     = freqs >= min_freq
    f_pos    = freqs[mask]
    p_pos    = power[mask]
    orig_idx = np.where(mask)[0]
    if len(p_pos) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    prom_thresh = 0.01 * np.max(p_pos)
    pks, _      = find_peaks(p_pos, prominence=prom_thresh)
    if len(pks) == 0:
        pks = np.array([np.argmax(p_pos)])
    pks_sorted = pks[np.argsort(p_pos[pks])[::-1]]
    pks_top    = pks_sorted[:N_peaks]
    omega_bin      = f_pos[pks_top]
    peak_powers    = p_pos[pks_top]
    d_omega        = freqs[1] - freqs[0]
    omega_parab    = np.array([quadratic_interpolate_peak(freqs, power, orig_idx[pk])
                               for pk in pks_top])
    omega_centroid = np.array([centroid_frequency(freqs, power, orig_idx[pk], centroid_bins)
                               for pk in pks_top])
    omega_gauss = []
    for pk_orig, omega_cen in zip(orig_idx[pks_top], omega_centroid):
        g = gaussian_log_fit(freqs_hann, power_hann, pk_orig, gauss_bins)
        window_lo = omega_cen - centroid_bins * d_omega
        window_hi = omega_cen + centroid_bins * d_omega
        if g is not None and window_lo <= g <= window_hi:
            omega_gauss.append(g)
        else:
            omega_gauss.append(omega_cen)
    omega_gauss = np.array(omega_gauss)
    return omega_bin, omega_parab, omega_centroid, omega_gauss, peak_powers

def select_physical_peak(omega_centroid, peak_powers, omega_evp,
                         evp_frac_window, min_power_frac):
    if len(omega_centroid) == 0:
        return None, False
    power_threshold = min_power_frac * np.max(peak_powers)
    power_ok        = peak_powers >= power_threshold
    evp_ok          = np.abs(omega_centroid - omega_evp) / (np.abs(omega_evp) + 1e-30) <= evp_frac_window
    candidates      = np.where(power_ok & evp_ok)[0]
    if len(candidates) > 0:
        dists     = np.abs(omega_centroid[candidates] - omega_evp)
        best      = candidates[np.argmin(dists)]
        evp_match = True
    else:
        best      = 0
        evp_match = False
    return best, evp_match

def flag_harmonics(omega_centroid, selected_idx, harmonic_frac, n_harmonics=3):
    omega_sel = omega_centroid[selected_idx]
    flags     = np.zeros(len(omega_centroid), dtype=bool)
    for pi in range(len(omega_centroid)):
        if pi == selected_idx:
            continue
        for n in range(2, n_harmonics + 1):
            if np.abs(omega_centroid[pi] - n * omega_sel) / (n * omega_sel + 1e-30) <= harmonic_frac:
                flags[pi] = True
                break
    return flags

def fit_amplitude_phase(t, y, omega):
    A_mat           = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    c1, c2 = coeffs
    A   = np.sqrt(c1**2 + c2**2)
    phi = np.arctan2(c2, c1)
    return A, phi, A_mat @ coeffs

def estimate_amplitude_extrema(y):
    """
    Estimate signal amplitude as the mean of absolute values of local extrema.
    Robust to beat patterns and harmonic contamination.
    Uses both maxima and minima.
    """
    prom_thresh = 0.01 * np.max(np.abs(y))
    pks_pos, _  = find_peaks( y, prominence=prom_thresh)
    pks_neg, _  = find_peaks(-y, prominence=prom_thresh)
    extrema_vals = np.concatenate([y[pks_pos], -y[pks_neg]])
    if len(extrema_vals) == 0:
        return np.max(np.abs(y))   # fallback
    return np.mean(extrema_vals)

### load processed projection file ###
logger.info("loading: " + processed_file_str)
processed = np.load(processed_file_str, allow_pickle=True)[()]

tw           = processed['tw']
idxs_include = processed['idxs_include']
projdot_c    = processed['projdot_c']
projdot_s    = processed['projdot_s']
projdot      = processed['projdot']
evals_re     = np.copy(processed['evals_re'])
evals_im     = np.copy(processed['evals_im'])
drifts       = np.copy(processed['drifts'])

nidxs   = len(idxs_include)
nw      = len(tw)
dt      = tw[1] - tw[0]
d_omega = 2 * np.pi / (tw[-1] - tw[0])

print(f"nidxs={nidxs}, nw={nw}, dt={dt:.5f}, T_window={tw[-1]-tw[0]:.3f}")
print(f"FFT frequency resolution: delta_omega = {d_omega:.4f} rad/time")

hann = np.hanning(nw)

### output arrays ###
fft_peak_freqs_bin      = np.zeros((nidxs, N_peaks))
fft_peak_freqs_parab    = np.zeros((nidxs, N_peaks))
fft_peak_freqs_centroid = np.zeros((nidxs, N_peaks))
fft_peak_freqs_gauss    = np.zeros((nidxs, N_peaks))
fft_peak_powers         = np.zeros((nidxs, N_peaks))
fft_peak_amps           = np.zeros((nidxs, N_peaks))
fft_peak_phases         = np.zeros((nidxs, N_peaks))
fft_peak_harmonic_flag  = np.zeros((nidxs, N_peaks), dtype=bool)
fft_selected_idx        = np.zeros(nidxs, dtype=int)
fft_evp_match           = np.zeros(nidxs, dtype=bool)

freqs_full, _  = real_fft_spectrum(projdot[0,:], dt, window=None)
fft_power_full = np.zeros((nidxs, len(freqs_full)))

projdot_fit        = np.zeros((nidxs, nw))
projdot_fit_params = np.zeros((nidxs, 3))   # A_sinusoid, omega_selected, phi
projdot_rel_err    = np.zeros((nidxs, nw))
projdot_amp_extrema = np.zeros(nidxs)       # primary amplitude estimate

### main loop ###
logger.info('entering FFT fitting loop')
for i, idx in enumerate(idxs_include):
    omega_evp = np.abs(evals_re[idx])
    print(f'\ni={i}, idx={idx},  eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  '
          f'drift={drifts[idx]:.3e},  omega_evp={omega_evp:.4f}')

    # extrema-based amplitude estimate (primary)
    amp_extrema = estimate_amplitude_extrema(projdot[i,:])
    projdot_amp_extrema[i] = amp_extrema
    print(f"  Extrema amplitude estimate: {amp_extrema:.4f}")

    # FFT
    freqs, fft_y     = real_fft_spectrum(projdot[i,:], dt, window=None)
    power            = np.abs(fft_y)**2
    power_norm       = power / (np.max(power) + 1e-30)
    fft_power_full[i,:] = power_norm

    freqs_h, fft_yh  = real_fft_spectrum(projdot[i,:], dt, window=hann)
    power_h          = np.abs(fft_yh)**2
    power_norm_h     = power_h / (np.max(power_h) + 1e-30)

    # find top N peaks
    (omega_bin, omega_parab, omega_centroid, omega_gauss, peak_powers) = find_top_peaks(
        freqs, power_norm, freqs_h, power_norm_h,
        N_peaks, min_freq, centroid_bins, gauss_bins
    )

    fft_peak_freqs_bin[i,      :len(omega_bin)]      = omega_bin
    fft_peak_freqs_parab[i,    :len(omega_parab)]    = omega_parab
    fft_peak_freqs_centroid[i, :len(omega_centroid)] = omega_centroid
    fft_peak_freqs_gauss[i,    :len(omega_gauss)]    = omega_gauss
    fft_peak_powers[i,         :len(peak_powers)]    = peak_powers

    # select physical peak using EVP proximity
    sel_idx, evp_match = select_physical_peak(
        omega_centroid, peak_powers, omega_evp,
        evp_frac_window, min_power_frac
    )
    fft_selected_idx[i] = sel_idx if sel_idx is not None else 0
    fft_evp_match[i]    = evp_match

    # flag harmonics
    if sel_idx is not None and len(omega_centroid) > 0:
        harm_flags = flag_harmonics(omega_centroid, sel_idx, harmonic_frac)
        fft_peak_harmonic_flag[i, :len(omega_centroid)] = harm_flags
    else:
        harm_flags = np.zeros(len(omega_centroid), dtype=bool)

    # amplitude/phase fit for all peaks
    print(f"  Top {len(omega_bin)} peaks  [selected={sel_idx}, evp_match={evp_match}]:")
    for pi in range(len(omega_bin)):
        omega_fit = omega_centroid[pi]
        A_fit, phi_fit, _ = fit_amplitude_phase(tw, projdot[i,:], omega_fit)
        fft_peak_amps[i,   pi] = A_fit
        fft_peak_phases[i, pi] = phi_fit
        harm_str = '  [HARMONIC?]' if harm_flags[pi] else ''
        sel_str  = '  <-- SELECTED' if pi == sel_idx else ''
        print(f"    peak {pi}: bin={omega_bin[pi]:.4f}  centroid={omega_centroid[pi]:.4f}  "
              f"power={peak_powers[pi]:.4f}  A_fit={A_fit:.4f}{harm_str}{sel_str}")

    # dominant fit using selected peak
    if sel_idx is not None and len(omega_centroid) > 0:
        omega_dom               = omega_centroid[sel_idx]
        A_dom, phi_dom, fit_dom = fit_amplitude_phase(tw, projdot[i,:], omega_dom)
        projdot_fit[i,:]        = fit_dom
        projdot_fit_params[i,:] = np.array([A_dom, omega_dom, phi_dom])
        rel_err                 = np.abs(projdot[i,:] - fit_dom) / (np.abs(fit_dom) + 1e-14)
        projdot_rel_err[i,:]    = rel_err
        print(f"  Selected fit: A_sinusoid={A_dom:.4f},  A_extrema={amp_extrema:.4f},  "
              f"omega={omega_dom:.4f},  phi={np.degrees(phi_dom):.1f} deg,  "
              f"mean_rel_err={np.mean(rel_err):.3f}")
        print(f"  EVP prediction: {omega_evp:.4f},  "
              f"fractional diff: {abs(omega_dom - omega_evp)/omega_evp:.3f}")
    else:
        print(f"  i={i}: no peaks found above min_freq={min_freq}")

    ### optional figure ###
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)

        # panel 1: time series and fit
        ax = axes[0]
        ax.plot(tw, projdot[i,:],  color='C0', lw=1.2, label='projection signal')
        #ax.plot(tw, projdot[i,:], '-o',  color='C0', lw=1.2, label='projdot (c+s)')
        ax.plot(tw, projdot_c[i,:], color='C6', lw=0.8, alpha=0.7, label=r'$\cos$ part')
        ax.plot(tw, projdot_s[i,:], color='C7', lw=0.8, alpha=0.7, label=r'$\sin$ part')
        if sel_idx is not None and len(omega_centroid) > 0:
            ax.plot(tw, projdot_fit[i,:], color='k', lw=1.2, ls='--',
                    label=f'sinusoid fit (omega={omega_dom:.4f})')
        ax.axhline( amp_extrema, color='gray', ls=':', lw=0.8,
                    label=f'extrema amp = {amp_extrema:.4f}')
        ax.axhline(-amp_extrema, color='gray', ls=':', lw=0.8)
        ax.set_xlabel('time')
        ax.set_ylabel('projection amplitude')
        ax.set_title(f'Mode i={i}, idx={idx}:  '
                     f'eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  '
                     f'drift={drifts[idx]:.2e}')
        ax.legend(fontsize=8)

        # panel 2: full power spectrum
        ax = axes[1]
        ax.plot(freqs, power_norm,     color='C0', lw=1.0, label='spectrum')
        #ax.plot(freqs_h, power_norm_h, color='C0', lw=0.8, ls=':', alpha=0.5,
        #        label='Hann-windowed')
        for pi in range(len(omega_bin)):
            c       = f'C{pi+1}'
            is_sel  = (pi == sel_idx)
            is_harm = harm_flags[pi] if pi < len(harm_flags) else False
            lw      = 2.0 if is_sel else 0.8
            ls      = '-' if is_sel else ('--' if is_harm else ':')
            label   = (f'pk{pi}: cen={omega_centroid[pi]:.3f}'
                       + (' [intrinsic?]' if is_sel else '')
                       + (' [harmonic?]' if is_harm else ''))
            if is_sel: 
                cpi = c
            ax.axvline(omega_centroid[pi], color=c, ls=ls, lw=lw, label=label)
        ax.axvline(omega_evp, color='gray', ls='--', lw=1.0,
                   label=f'EVP pred: {omega_evp:.3f}')
        ax.set_xlabel('omega (rad/time)')
        ax.set_ylabel('normalised power')
        ax.set_title('Power spectrum — centroid peaks marked')
        if len(omega_bin) > 0:
            ax.set_xlim([0, max(3 * np.max(omega_bin), min_freq * 2)])
        ax.legend(fontsize=7)

        # panel 3: zoom on selected peak
        ax = axes[2]
        if sel_idx is not None and len(omega_bin) > 0:
            zoom_hw = 5 * d_omega
            zmask   = np.abs(freqs - omega_bin[sel_idx]) <= zoom_hw
            ax.plot(freqs[zmask], power_norm[zmask], color='C0', lw=1.2,
                    marker='o', ms=4, label='spectrum')
            #ax.plot(freqs_h[zmask], power_norm_h[zmask], color='C0', lw=0.8,
            #        ls=':', alpha=0.6, label='Hann-windowed')
            #ax.axvline(omega_bin[sel_idx],      color='C1', ls=':',  lw=1.0,
            #           label=f'bin={omega_bin[sel_idx]:.4f}')
            #ax.axvline(omega_parab[sel_idx],    color='C2', ls='--', lw=1.0,
            #           label=f'parab={omega_parab[sel_idx]:.4f}')
            ax.axvline(omega_centroid[sel_idx], color=cpi, ls='-',  lw=2.0,
                       label=f'centroid fit={omega_centroid[sel_idx]:.4f}')
            ax.axvline(omega_evp, color='gray', ls='--', lw=1.0,
                       label=f'EVP={omega_evp:.4f}')
        ax.set_xlabel('omega (rad/time)')
        ax.set_ylabel('normalised power')
        ax.set_title('Zoom on selected peak')
        ax.legend(fontsize=8)

        # panel 4: normalized fit residual
        ax = axes[3]
        if sel_idx is not None and len(omega_centroid) > 0 and amp_extrema > 0:
            residual_norm = (projdot[i,:] - projdot_fit[i,:]) / amp_extrema
            ax.plot(tw, residual_norm, color=cpi, lw=0.8)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel('relative error')
        ax.set_xlabel('time')
        ax.set_title('Normalized fit residual  '
                     '(beat pattern visible as slow modulation)')

        #fig.suptitle(f'FFT fitting — mode i={i}  [{output_suffix}]', fontsize=10)
        fig.suptitle(f'FFT fitting')
        figname = f'fft_fit_mode_i{i:02d}_{output_suffix}.png'
        fig.savefig(figname, dpi=200)
        plt.close(fig)
        print(f"  Figure saved: {figname}")

### sort by dominant extrema amplitude, largest first ###
amp_order = np.argsort(projdot_amp_extrema)[::-1]
print(f"\nAmplitude sort order (by extrema amp): {amp_order}")

def reorder(arr, order):
    return arr[order] if arr.ndim == 1 else arr[order, :]

projdot_fit             = reorder(projdot_fit,             amp_order)
projdot_fit_params      = reorder(projdot_fit_params,      amp_order)
projdot_rel_err         = reorder(projdot_rel_err,         amp_order)
projdot_amp_extrema     = projdot_amp_extrema[amp_order]
fft_peak_freqs_bin      = reorder(fft_peak_freqs_bin,      amp_order)
fft_peak_freqs_parab    = reorder(fft_peak_freqs_parab,    amp_order)
fft_peak_freqs_centroid = reorder(fft_peak_freqs_centroid, amp_order)
fft_peak_freqs_gauss    = reorder(fft_peak_freqs_gauss,    amp_order)
fft_peak_powers         = reorder(fft_peak_powers,         amp_order)
fft_peak_amps           = reorder(fft_peak_amps,           amp_order)
fft_peak_phases         = reorder(fft_peak_phases,         amp_order)
fft_peak_harmonic_flag  = reorder(fft_peak_harmonic_flag,  amp_order)
fft_power_full          = reorder(fft_power_full,          amp_order)
fft_selected_idx        = fft_selected_idx[amp_order]
fft_evp_match           = fft_evp_match[amp_order]
evals_re                = evals_re[amp_order]
evals_im                = evals_im[amp_order]
drifts                  = drifts[amp_order]

### assemble and save output ###
fit_results = {}
fit_results.update(processed)

for key in ['evals_re', 'evals_im', 'drifts',
            'projl2_fit', 'projl2_fit_params', 'projl2_rel_err',
            'mpm_freqs',  'mpm_decays']:
    fit_results.pop(key, None)

for key in ['projdot_c', 'projdot_s', 'projdot', 'projdot_stats',
            'projl2', 'projl2_stats', 'ivpl2', 'ivpl2_stats', 'evpl2']:
    if key in fit_results:
        fit_results[key] = reorder(fit_results[key], amp_order)

fit_results['evals_re'] = evals_re
fit_results['evals_im'] = evals_im
fit_results['drifts']   = drifts

fit_results['projdot_fit']          = projdot_fit
fit_results['projdot_fit_params']   = projdot_fit_params    # A_sinusoid, omega_selected, phi
fit_results['projdot_rel_err']      = projdot_rel_err
fit_results['projdot_amp_extrema']  = projdot_amp_extrema   # primary amplitude estimate

fit_results['fft_peak_freqs']          = fft_peak_freqs_centroid
fit_results['fft_peak_freqs_bin']      = fft_peak_freqs_bin
fit_results['fft_peak_freqs_parab']    = fft_peak_freqs_parab
fit_results['fft_peak_freqs_centroid'] = fft_peak_freqs_centroid
fit_results['fft_peak_freqs_gauss']    = fft_peak_freqs_gauss
fit_results['fft_peak_powers']         = fft_peak_powers
fit_results['fft_peak_amps']           = fft_peak_amps
fit_results['fft_peak_phases']         = fft_peak_phases
fit_results['fft_peak_harmonic_flag']  = fft_peak_harmonic_flag
fit_results['fft_selected_idx']        = fft_selected_idx
fit_results['fft_evp_match']           = fft_evp_match
fit_results['fft_power_full']          = fft_power_full
fit_results['fft_freqs_full']          = freqs_full

out_path = output_prefix + '_' + output_suffix + '.npy'
print(f"\nSaving output as: {out_path}")
np.save(out_path, fit_results)

"""
Fit oscillation frequencies of Rossby mode projection amplitudes using FFT,
loading from pre-processed projection output.

Replaces the MPM-based process_fit_rossby_projections_mpm.py.

Uses projdot = projdot_c + projdot_s, the total real-valued projection
coefficient, as the signal to FFT. np.fft.rfft is used since projdot is
real-valued.

Three sub-bin frequency estimates are computed for each peak:
  1. omega_bin    : raw FFT bin center (reference)
  2. omega_parab  : 3-point quadratic interpolation (as before, kept for reference)
  3. omega_centroid: power-weighted centroid over a window around the peak
  4. omega_gauss  : Gaussian fit to log-power around the peak, optionally with
                    a Hann window applied to the time series before FFT

The Hann-windowed FFT is computed separately and used only for the Gaussian
fit, so the centroid and parabolic estimates still use the un-windowed FFT.
The amplitude/phase least-squares fit uses omega_gauss as the primary
frequency estimate, with omega_centroid as a fallback if the Gaussian fit
fails.

Usage:
    process_fit_rossby_projections_fft.py <processed_file> [options]

Options:
    --output=<str>          prefix for the output file [default: processed_rossby_projection_fft]
    --N_peaks=<int>         number of spectral peaks to retain per mode [default: 5]
    --min_freq=<float>      minimum frequency to consider (rad/time) [default: 0.5]
    --centroid_bins=<int>   half-width in bins for centroid window [default: 4]
    --gauss_bins=<int>      half-width in bins for Gaussian fit window [default: 6]
    --plot=<bool>           if True, save a summary figure per mode [default: True]
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

processed_file_str = args['<processed_file>']
output             = args['--output']
N_peaks            = int(args['--N_peaks'])
min_freq           = float(args['--min_freq'])
centroid_bins      = int(args['--centroid_bins'])
gauss_bins         = int(args['--gauss_bins'])
plot               = eval(args['--plot'])

output_suffix = processed_file_str.split('processed_rossby_projection_')[1].split('.')[0].split('/')[0]
output_prefix = output

### helper functions ###

def real_fft_spectrum(y, dt, window=None):
    """
    Real FFT of a real-valued signal y, optionally with a window applied.
    Returns positive angular frequencies (rad/time) and power spectrum.
    """
    N = len(y)
    if window is not None:
        y = y * window
        # normalise so amplitude is preserved
        y = y * N / np.sum(window)
    fft_y = np.fft.rfft(y) / N
    freqs  = np.fft.rfftfreq(N, d=dt) * 2 * np.pi
    return freqs, fft_y

def quadratic_interpolate_peak(freqs, power, peak_idx):
    """3-point quadratic interpolation. Kept for reference."""
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
    """
    Power-weighted centroid frequency over a window of +/- half_width bins
    around peak_idx. Naturally handles asymmetric peaks.
    """
    N     = len(freqs)
    lo    = max(0, peak_idx - half_width)
    hi    = min(N - 1, peak_idx + half_width)
    w     = power[lo:hi+1]
    f     = freqs[lo:hi+1]
    denom = np.sum(w)
    if denom < 1e-30:
        return freqs[peak_idx]
    return np.sum(f * w) / denom

def gaussian_log_fit(freqs, power, peak_idx, half_width):
    """
    Fit a Gaussian to the log-power spectrum in a window around the peak.
    Returns the fitted center frequency, or None if the fit fails.

    Fitting in log-power space:
        log(P) ~ log(A) - (omega - mu)^2 / (2*sigma^2)
    which is a parabola in log-power — this is more robust to asymmetry
    than fitting in linear power space.
    """
    N  = len(freqs)
    lo = max(0, peak_idx - half_width)
    hi = min(N - 1, peak_idx + half_width)
    f  = freqs[lo:hi+1]
    p  = power[lo:hi+1]

    # avoid log of zero
    p = np.where(p > 1e-30 * np.max(p), p, 1e-30 * np.max(p))
    log_p = np.log(p)

    def log_gauss(x, log_A, mu, sigma):
        return log_A - 0.5 * ((x - mu) / sigma)**2

    try:
        p0    = [log_p[np.argmax(log_p)], freqs[peak_idx], (freqs[hi] - freqs[lo]) / 4]
        bounds = ([-np.inf, freqs[lo], 1e-6], [np.inf, freqs[hi], freqs[hi]-freqs[lo]])
        popt, _ = curve_fit(log_gauss, f, log_p, p0=p0, bounds=bounds, maxfev=2000)
        return popt[1]   # mu = fitted center frequency
    except Exception:
        return None

def find_top_peaks(freqs, power, freqs_hann, power_hann,
                   N_peaks, min_freq, centroid_bins, gauss_bins):
    """
    Find top N_peaks peaks above min_freq in the un-windowed power spectrum.
    For each peak compute:
      - omega_bin      : bin center
      - omega_parab    : 3-point quadratic interpolation (un-windowed)
      - omega_centroid : power-weighted centroid (un-windowed)
      - omega_gauss    : Gaussian fit to log-power (Hann-windowed)
    Returns lists of each, plus normalised powers.
    """
    mask     = freqs >= min_freq
    f_pos    = freqs[mask]
    p_pos    = power[mask]
    orig_idx = np.where(mask)[0]

    if len(p_pos) == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty

    prom_thresh = 0.01 * np.max(p_pos)
    pks, _      = find_peaks(p_pos, prominence=prom_thresh)
    if len(pks) == 0:
        pks = np.array([np.argmax(p_pos)])

    pks_sorted = pks[np.argsort(p_pos[pks])[::-1]]
    pks_top    = pks_sorted[:N_peaks]

    omega_bin      = f_pos[pks_top]
    peak_powers    = p_pos[pks_top]
    omega_parab    = np.array([quadratic_interpolate_peak(freqs, power, orig_idx[pk])
                               for pk in pks_top])
    omega_centroid = np.array([centroid_frequency(freqs, power, orig_idx[pk], centroid_bins)
                               for pk in pks_top])

    # Gaussian fit on Hann-windowed spectrum
    # find corresponding peak index in Hann spectrum (nearest bin)
    omega_gauss = []
    for pk_orig in orig_idx[pks_top]:
        g = gaussian_log_fit(freqs_hann, power_hann, pk_orig, gauss_bins)
        omega_gauss.append(g if g is not None else freqs[pk_orig])
    omega_gauss = np.array(omega_gauss)

    return omega_bin, omega_parab, omega_centroid, omega_gauss, peak_powers

def fit_amplitude_phase(t, y, omega):
    """Least-squares fit of y = A*sin(omega*t + phi)."""
    A_mat           = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    c1, c2 = coeffs
    A   = np.sqrt(c1**2 + c2**2)
    phi = np.arctan2(c2, c1)
    return A, phi, A_mat @ coeffs

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

# Hann window (computed once)
hann = np.hanning(nw)

### output arrays ###
fft_peak_freqs_bin      = np.zeros((nidxs, N_peaks))
fft_peak_freqs_parab    = np.zeros((nidxs, N_peaks))
fft_peak_freqs_centroid = np.zeros((nidxs, N_peaks))
fft_peak_freqs_gauss    = np.zeros((nidxs, N_peaks))
fft_peak_powers         = np.zeros((nidxs, N_peaks))
fft_peak_amps           = np.zeros((nidxs, N_peaks))
fft_peak_phases         = np.zeros((nidxs, N_peaks))

projdot_fit        = np.zeros((nidxs, nw))
projdot_fit_params = np.zeros((nidxs, 3))   # A, omega_gauss (primary), phi
projdot_rel_err    = np.zeros((nidxs, nw))

### main loop ###
logger.info('entering FFT fitting loop')
for i, idx in enumerate(idxs_include):
    print(f'\ni={i}, idx={idx},  eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  drift={drifts[idx]:.3e}')

    # un-windowed FFT
    freqs, fft_y      = real_fft_spectrum(projdot[i,:], dt, window=None)
    power             = np.abs(fft_y)**2
    power_norm        = power / (np.max(power) + 1e-30)

    # Hann-windowed FFT (for Gaussian fit only)
    freqs_h, fft_yh   = real_fft_spectrum(projdot[i,:], dt, window=hann)
    power_h           = np.abs(fft_yh)**2
    power_norm_h      = power_h / (np.max(power_h) + 1e-30)

    # find peaks and compute all frequency estimates
    (omega_bin, omega_parab, omega_centroid, omega_gauss, peak_powers) = find_top_peaks(
        freqs, power_norm, freqs_h, power_norm_h,
        N_peaks, min_freq, centroid_bins, gauss_bins
    )

    fft_peak_freqs_bin[i,      :len(omega_bin)]      = omega_bin
    fft_peak_freqs_parab[i,    :len(omega_parab)]    = omega_parab
    fft_peak_freqs_centroid[i, :len(omega_centroid)] = omega_centroid
    fft_peak_freqs_gauss[i,    :len(omega_gauss)]    = omega_gauss
    fft_peak_powers[i,         :len(peak_powers)]    = peak_powers

    print(f"  Top {len(omega_bin)} peaks (rad/time):")
    for pi in range(len(omega_bin)):
        # use Gaussian estimate as primary frequency for amplitude/phase fit
        omega_fit = omega_gauss[pi]
        A_fit, phi_fit, _ = fit_amplitude_phase(tw, projdot[i,:], omega_fit)
        fft_peak_amps[i,   pi] = A_fit
        fft_peak_phases[i, pi] = phi_fit
        print(f"    peak {pi}: bin={omega_bin[pi]:.4f}  parab={omega_parab[pi]:.4f}  "
              f"centroid={omega_centroid[pi]:.4f}  gauss={omega_gauss[pi]:.4f}  "
              f"power={peak_powers[pi]:.4f}  A={A_fit:.4f}")

    # dominant peak fit
    if len(omega_gauss) > 0:
        omega_dom               = omega_gauss[0]
        A_dom, phi_dom, fit_dom = fit_amplitude_phase(tw, projdot[i,:], omega_dom)
        projdot_fit[i,:]        = fit_dom
        projdot_fit_params[i,:] = np.array([A_dom, omega_dom, phi_dom])
        rel_err                 = np.abs(projdot[i,:] - fit_dom) / (np.abs(fit_dom) + 1e-14)
        projdot_rel_err[i,:]    = rel_err
        print(f"  Dominant fit (gauss): A={A_dom:.4f},  omega={omega_dom:.4f},  "
              f"phi={np.degrees(phi_dom):.1f} deg,  mean_rel_err={np.mean(rel_err):.3f}")
    else:
        print(f"  i={i}: no peaks found above min_freq={min_freq}")

    ### optional figure ###
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)

        # panel 1: projdot components and dominant fit
        ax = axes[0]
        ax.plot(tw, projdot[i,:],   color='C0', lw=1.2, label='projdot (c+s)')
        ax.plot(tw, projdot_c[i,:], color='C2', lw=0.8, alpha=0.7, label='projdot_c')
        ax.plot(tw, projdot_s[i,:], color='C3', lw=0.8, alpha=0.7, label='projdot_s')
        if len(omega_gauss) > 0:
            ax.plot(tw, projdot_fit[i,:], color='k', lw=1.2, ls='--',
                    label=f'dominant fit (gauss omega={omega_dom:.4f})')
        ax.set_xlabel('time')
        ax.set_ylabel('projection amplitude')
        ax.set_title(f'Mode i={i}, idx={idx}:  '
                     f'eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  drift={drifts[idx]:.2e}')
        ax.legend(fontsize=8)

        # panel 2: un-windowed power spectrum with all frequency estimates
        ax = axes[1]
        colors = [f'C{k+1}' for k in range(len(omega_bin))]
        for pi in range(len(omega_bin)):
            c = colors[pi]
            ax.axvline(omega_bin[pi],      color=c, ls=':',  lw=0.8)
            ax.axvline(omega_parab[pi],    color=c, ls='--', lw=0.8)
            ax.axvline(omega_centroid[pi], color=c, ls='-',  lw=1.0)
            ax.axvline(omega_gauss[pi],    color=c, ls='-',  lw=1.5,
                       label=f'pk{pi}: bin={omega_bin[pi]:.3f} par={omega_parab[pi]:.3f} '
                             f'cen={omega_centroid[pi]:.3f} gau={omega_gauss[pi]:.3f}')
        ax.plot(freqs, power_norm, color='C0', lw=1.0, label='un-windowed')
        ax.plot(freqs_h, power_norm_h, color='C0', lw=0.8, ls=':', alpha=0.5, label='Hann-windowed')
        ax.set_xlabel('omega (rad/time)')
        ax.set_ylabel('normalised power')
        ax.set_title('Power spectrum — thin:parab  medium:centroid  thick:gauss  (dotted=bin)')
        if len(omega_bin) > 0:
            ax.set_xlim([0, max(3 * omega_bin[0], min_freq * 2)])
        ax.legend(fontsize=6)

        # panel 3: zoom on dominant peak to see interpolation differences
        ax = axes[2]
        if len(omega_bin) > 0:
            zoom_hw = 5 * d_omega
            zmask   = np.abs(freqs - omega_bin[0]) <= zoom_hw
            ax.plot(freqs[zmask], power_norm[zmask], color='C0', lw=1.2, marker='o',
                    ms=4, label='un-windowed bins')
            ax.plot(freqs_h[zmask], power_norm_h[zmask], color='C0', lw=0.8, ls=':',
                    alpha=0.6, label='Hann-windowed')
            ax.axvline(omega_bin[0],      color='C1', ls=':',  lw=1.0, label=f'bin={omega_bin[0]:.4f}')
            ax.axvline(omega_parab[0],    color='C2', ls='--', lw=1.0, label=f'parab={omega_parab[0]:.4f}')
            ax.axvline(omega_centroid[0], color='C3', ls='-',  lw=1.2, label=f'centroid={omega_centroid[0]:.4f}')
            ax.axvline(omega_gauss[0],    color='C4', ls='-',  lw=1.5, label=f'gauss={omega_gauss[0]:.4f}')
        ax.set_xlabel('omega (rad/time)')
        ax.set_ylabel('normalised power')
        ax.set_title('Zoom on dominant peak — all four frequency estimates')
        ax.legend(fontsize=8)

        # panel 4: instantaneous phase
        ax = axes[3]
        inst_phase = np.degrees(np.arctan2(projdot_s[i,:], projdot_c[i,:]))
        ax.plot(tw, inst_phase, color='C3', lw=0.8)
        ax.set_xlabel('time')
        ax.set_ylabel('instantaneous phase (deg)')
        ax.set_title('Phase: atan2(projdot_s, projdot_c)  —  compare with phi_CPC(t)')

        fig.suptitle(f'FFT fitting — mode i={i}  [{output_suffix}]', fontsize=10)
        figname = f'fft_fit_mode_i{i:02d}_{output_suffix}.png'
        fig.savefig(figname, dpi=120)
        plt.close(fig)
        print(f"  Figure saved: {figname}")

### sort by dominant amplitude, largest first ###
amp_order = np.argsort(projdot_fit_params[:, 0])[::-1]
print(f"\nAmplitude sort order: {amp_order}")

def reorder(arr, order):
    return arr[order] if arr.ndim == 1 else arr[order, :]

projdot_fit             = reorder(projdot_fit,             amp_order)
projdot_fit_params      = reorder(projdot_fit_params,      amp_order)
projdot_rel_err         = reorder(projdot_rel_err,         amp_order)
fft_peak_freqs_bin      = reorder(fft_peak_freqs_bin,      amp_order)
fft_peak_freqs_parab    = reorder(fft_peak_freqs_parab,    amp_order)
fft_peak_freqs_centroid = reorder(fft_peak_freqs_centroid, amp_order)
fft_peak_freqs_gauss    = reorder(fft_peak_freqs_gauss,    amp_order)
fft_peak_powers         = reorder(fft_peak_powers,         amp_order)
fft_peak_amps           = reorder(fft_peak_amps,           amp_order)
fft_peak_phases         = reorder(fft_peak_phases,         amp_order)
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

fit_results['projdot_fit']        = projdot_fit
fit_results['projdot_fit_params'] = projdot_fit_params    # A, omega_gauss, phi
fit_results['projdot_rel_err']    = projdot_rel_err

# all four frequency estimates stored for reference
fit_results['fft_peak_freqs']         = fft_peak_freqs_gauss    # primary
fit_results['fft_peak_freqs_bin']     = fft_peak_freqs_bin
fit_results['fft_peak_freqs_parab']   = fft_peak_freqs_parab
fit_results['fft_peak_freqs_centroid']= fft_peak_freqs_centroid
fit_results['fft_peak_freqs_gauss']   = fft_peak_freqs_gauss
fit_results['fft_peak_powers']        = fft_peak_powers
fit_results['fft_peak_amps']          = fft_peak_amps
fit_results['fft_peak_phases']        = fft_peak_phases

out_path = output_prefix + '_' + output_suffix + '.npy'
print(f"\nSaving output as: {out_path}")
np.save(out_path, fit_results)

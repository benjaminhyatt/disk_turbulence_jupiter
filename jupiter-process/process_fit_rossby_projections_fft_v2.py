"""
Fit oscillation frequencies of Rossby mode projection amplitudes using FFT,
loading from pre-processed projection output.

Replaces the MPM-based process_fit_rossby_projections_mpm.py.

Uses projdot = projdot_c + projdot_s, the total real-valued projection
coefficient, as the signal to FFT. projdot_c and projdot_s are the cosine
and sine quadrature components of the projection respectively, but their
relative amplitudes depend on the arbitrary complex phase of the EVP
eigenmode, so only their sum is physically meaningful as a single signal.

np.fft.rfft is used since projdot is real-valued.

Sub-bin frequency accuracy is achieved via quadratic (parabolic) interpolation
on the three FFT power values surrounding each detected peak. This is important
because the frequency resolution 2*pi/T_window is coarse relative to the
differences between candidate frequencies, and a bin-center estimate will
accumulate visible phase error over even short time windows.

Usage:
    process_fit_rossby_projections_fft.py <processed_file> [options]

Options:
    --output=<str>      prefix for the output file [default: processed_rossby_projection_fft]
    --N_peaks=<int>     number of spectral peaks to retain per mode [default: 5]
    --min_freq=<float>  minimum frequency to consider (rad/time) [default: 0.5]
    --plot=<bool>       if True, save a summary figure per mode [default: True]
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from docopt import docopt
from scipy.signal import find_peaks

### read in options ###
args = docopt(__doc__)
logger.info("args read in")
print(args)

processed_file_str = args['<processed_file>']
output             = args['--output']
N_peaks            = int(args['--N_peaks'])
min_freq           = float(args['--min_freq'])
plot               = eval(args['--plot'])

output_suffix = processed_file_str.split('processed_rossby_projection_')[1].split('.')[0].split('/')[0]
output_prefix = output

### helper functions ###

def real_fft_spectrum(y, dt):
    """
    Real FFT of a real-valued signal y.
    Returns positive angular frequencies (rad/time) and complex FFT coefficients.
    Uses np.fft.rfft consistent with Dedalus' real Fourier basis convention.
    """
    N     = len(y)
    fft_y = np.fft.rfft(y) / N
    freqs = np.fft.rfftfreq(N, d=dt) * 2 * np.pi   # rad/time, positive only
    return freqs, fft_y

def quadratic_interpolate_peak(freqs, power, peak_idx):
    """
    Sub-bin frequency refinement via quadratic (parabolic) interpolation.
    Fits a parabola through the peak bin and its two immediate neighbors
    to estimate the true peak location to sub-bin precision.

    Formula:
        delta = (P[k+1] - P[k-1]) / (2 * (2*P[k] - P[k-1] - P[k+1]))
        omega_true = omega[k] + delta * d_omega

    Returns the interpolated frequency. Falls back to the bin-center value
    if the peak is at the edge of the spectrum.
    """
    k      = peak_idx
    d_omega = freqs[1] - freqs[0]   # uniform bin spacing

    if k == 0 or k == len(freqs) - 1:
        # edge case: can't interpolate, return bin center
        return freqs[k]

    Pk_m = power[k - 1]
    Pk   = power[k]
    Pk_p = power[k + 1]

    denom = 2 * (2 * Pk - Pk_m - Pk_p)
    if np.abs(denom) < 1e-30:
        return freqs[k]

    delta = (Pk_p - Pk_m) / denom
    return freqs[k] + delta * d_omega

def find_top_peaks(freqs, power, N_peaks, min_freq):
    """
    Find the top N_peaks peaks in the power spectrum above min_freq.
    Returns:
        peak_freqs_interp : sub-bin interpolated peak frequencies
        peak_freqs_bin    : bin-center peak frequencies (for reference)
        peak_powers       : normalised power at each bin-center peak
    """
    mask      = freqs >= min_freq
    freqs_pos = freqs[mask]
    power_pos = power[mask]
    orig_idx  = np.where(mask)[0]   # indices into original freqs/power arrays

    if len(power_pos) == 0:
        return np.array([]), np.array([]), np.array([])

    prominence_thresh = 0.01 * np.max(power_pos)
    pks, _            = find_peaks(power_pos, prominence=prominence_thresh)

    if len(pks) == 0:
        pks = np.array([np.argmax(power_pos)])

    pks_sorted = pks[np.argsort(power_pos[pks])[::-1]]
    pks_top    = pks_sorted[:N_peaks]

    peak_freqs_bin    = freqs_pos[pks_top]
    peak_powers       = power_pos[pks_top]

    # quadratic interpolation — operates on the full (unmasked) arrays
    # so that boundary handling is correct for peaks near min_freq
    peak_freqs_interp = np.array([
        quadratic_interpolate_peak(freqs, power, orig_idx[pk])
        for pk in pks_top
    ])

    return peak_freqs_interp, peak_freqs_bin, peak_powers

def fit_amplitude_phase(t, y, omega):
    """
    Least-squares fit of y = A*sin(omega*t + phi).
    Returns amplitude A, phase phi, and the fitted signal.
    """
    A_mat           = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    c1, c2          = coeffs
    A               = np.sqrt(c1**2 + c2**2)
    phi             = np.arctan2(c2, c1)
    return A, phi, A_mat @ coeffs

### load processed projection file ###
logger.info("loading: " + processed_file_str)
processed = np.load(processed_file_str, allow_pickle=True)[()]

tw           = processed['tw']
idxs_include = processed['idxs_include']
projdot_c    = processed['projdot_c']   # shape (nidxs, nw)
projdot_s    = processed['projdot_s']   # shape (nidxs, nw)
projdot      = processed['projdot']     # shape (nidxs, nw) -- the signal we FFT
evals_re     = np.copy(processed['evals_re'])
evals_im     = np.copy(processed['evals_im'])
drifts       = np.copy(processed['drifts'])

nidxs  = len(idxs_include)
nw     = len(tw)
dt     = tw[1] - tw[0]
d_omega = 2 * np.pi / (tw[-1] - tw[0])   # FFT frequency resolution

print(f"nidxs={nidxs}, nw={nw}, dt={dt:.5f}, T_window={tw[-1]-tw[0]:.3f}")
print(f"FFT frequency resolution: delta_omega = {d_omega:.4f} rad/time")

### output arrays ###
fft_peak_freqs_interp = np.zeros((nidxs, N_peaks))   # sub-bin interpolated
fft_peak_freqs_bin    = np.zeros((nidxs, N_peaks))   # bin-center (for reference)
fft_peak_powers       = np.zeros((nidxs, N_peaks))
fft_peak_amps         = np.zeros((nidxs, N_peaks))
fft_peak_phases       = np.zeros((nidxs, N_peaks))

projdot_fit        = np.zeros((nidxs, nw))
projdot_fit_params = np.zeros((nidxs, 3))   # A, omega_dom (interpolated), phi
projdot_rel_err    = np.zeros((nidxs, nw))

### main loop ###
logger.info('entering FFT fitting loop')
for i, idx in enumerate(idxs_include):
    print(f'\ni={i}, idx={idx},  eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  drift={drifts[idx]:.3e}')

    # FFT of the total real-valued projection signal
    freqs, fft_y = real_fft_spectrum(projdot[i, :], dt)
    power        = np.abs(fft_y)**2
    power_norm   = power / (np.max(power) + 1e-30)

    # find top peaks with sub-bin interpolation
    peak_freqs_interp, peak_freqs_bin, peak_powers = find_top_peaks(
        freqs, power_norm, N_peaks, min_freq
    )

    fft_peak_freqs_interp[i, :len(peak_freqs_interp)] = peak_freqs_interp
    fft_peak_freqs_bin[i,    :len(peak_freqs_bin)]    = peak_freqs_bin
    fft_peak_powers[i,       :len(peak_powers)]       = peak_powers

    print(f"  Top {len(peak_freqs_interp)} peaks (rad/time):")
    for pi, (pf_interp, pf_bin, pp) in enumerate(
            zip(peak_freqs_interp, peak_freqs_bin, peak_powers)):
        A_fit, phi_fit, _ = fit_amplitude_phase(tw, projdot[i, :], pf_interp)
        fft_peak_amps[i,   pi] = A_fit
        fft_peak_phases[i, pi] = phi_fit
        print(f"    peak {pi}: omega_bin={pf_bin:.4f},  omega_interp={pf_interp:.4f},  "
              f"norm_power={pp:.4f},  A={A_fit:.4f},  phi={np.degrees(phi_fit):.1f} deg")

    # dominant peak fit using interpolated frequency
    if len(peak_freqs_interp) > 0:
        omega_dom               = peak_freqs_interp[0]
        A_dom, phi_dom, fit_dom = fit_amplitude_phase(tw, projdot[i, :], omega_dom)
        projdot_fit[i, :]       = fit_dom
        projdot_fit_params[i,:] = np.array([A_dom, omega_dom, phi_dom])
        rel_err                 = np.abs(projdot[i,:] - fit_dom) / (np.abs(fit_dom) + 1e-14)
        projdot_rel_err[i, :]   = rel_err
        print(f"  Dominant fit: A={A_dom:.4f},  omega={omega_dom:.4f},  "
              f"phi={np.degrees(phi_dom):.1f} deg,  mean_rel_err={np.mean(rel_err):.3f}")
    else:
        print(f"  i={i}, idx={idx}: no peaks found above min_freq={min_freq}")

    ### optional figure ###
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

        # panel 1: projdot_c, projdot_s, projdot, dominant fit
        ax = axes[0]
        ax.plot(tw, projdot[i,:],   color='C0', lw=1.2, label='projdot (c+s)')
        ax.plot(tw, projdot_c[i,:], color='C2', lw=0.8, alpha=0.7, label='projdot_c')
        ax.plot(tw, projdot_s[i,:], color='C3', lw=0.8, alpha=0.7, label='projdot_s')
        if len(peak_freqs_interp) > 0:
            ax.plot(tw, projdot_fit[i,:], color='k', lw=1.2, ls='--',
                    label=f'dominant fit (omega={omega_dom:.4f})')
        ax.set_xlabel('time')
        ax.set_ylabel('projection amplitude')
        ax.set_title(f'Mode i={i}, idx={idx}:  '
                     f'eval={evals_re[idx]:.4f}+i{evals_im[idx]:.4f},  '
                     f'drift={drifts[idx]:.2e}')
        ax.legend(fontsize=8)

        # panel 2: power spectrum with bin-center and interpolated peaks marked
        ax = axes[1]
        ax.plot(freqs, power_norm, color='C0', lw=1.0)
        for pi, (pf_interp, pf_bin, pp) in enumerate(
                zip(peak_freqs_interp, peak_freqs_bin, peak_powers)):
            ax.axvline(pf_bin,    color=f'C{pi+1}', ls=':',  lw=0.8,
                       label=f'peak {pi} bin:    {pf_bin:.4f}')
            ax.axvline(pf_interp, color=f'C{pi+1}', ls='--', lw=1.0,
                       label=f'peak {pi} interp: {pf_interp:.4f}')
        ax.set_xlabel('omega (rad/time)')
        ax.set_ylabel('normalised power')
        ax.set_title('FFT power spectrum (rfft) — dotted=bin center, dashed=interpolated')
        if len(peak_freqs_interp) > 0:
            ax.set_xlim([0, max(3 * peak_freqs_bin[0], min_freq * 2)])
        ax.legend(fontsize=7)

        # panel 3: instantaneous phase from projdot_c and projdot_s
        ax = axes[2]
        inst_phase = np.degrees(np.arctan2(projdot_s[i,:], projdot_c[i,:]))
        ax.plot(tw, inst_phase, color='C3', lw=0.8)
        ax.set_xlabel('time')
        ax.set_ylabel('instantaneous phase (deg)')
        ax.set_title('Phase: atan2(projdot_s, projdot_c)  —  compare with phi_CPC(t)')

        fig.suptitle(f'FFT fitting summary — mode i={i}  [{output_suffix}]', fontsize=10)
        figname = f'fft_fit_mode_i{i:02d}_{output_suffix}.png'
        fig.savefig(figname, dpi=120)
        plt.close(fig)
        print(f"  Figure saved: {figname}")

### sort by dominant amplitude, largest first ###
amp_order = np.argsort(projdot_fit_params[:, 0])[::-1]
print(f"\nAmplitude sort order: {amp_order}")

def reorder(arr, order):
    return arr[order] if arr.ndim == 1 else arr[order, :]

projdot_fit           = reorder(projdot_fit,           amp_order)
projdot_fit_params    = reorder(projdot_fit_params,    amp_order)
projdot_rel_err       = reorder(projdot_rel_err,       amp_order)
fft_peak_freqs_interp = reorder(fft_peak_freqs_interp, amp_order)
fft_peak_freqs_bin    = reorder(fft_peak_freqs_bin,    amp_order)
fft_peak_powers       = reorder(fft_peak_powers,       amp_order)
fft_peak_amps         = reorder(fft_peak_amps,         amp_order)
fft_peak_phases       = reorder(fft_peak_phases,       amp_order)
evals_re              = evals_re[amp_order]
evals_im              = evals_im[amp_order]
drifts                = drifts[amp_order]

### assemble and save output ###
fit_results = {}
fit_results.update(processed)

# remove keys being replaced
for key in ['evals_re', 'evals_im', 'drifts',
            'projl2_fit', 'projl2_fit_params', 'projl2_rel_err',
            'mpm_freqs',  'mpm_decays']:
    fit_results.pop(key, None)

# reorder inherited arrays that depend on mode index
for key in ['projdot_c', 'projdot_s', 'projdot', 'projdot_stats',
            'projl2', 'projl2_stats', 'ivpl2', 'ivpl2_stats', 'evpl2']:
    if key in fit_results:
        fit_results[key] = reorder(fit_results[key], amp_order)

fit_results['evals_re'] = evals_re
fit_results['evals_im'] = evals_im
fit_results['drifts']   = drifts

fit_results['projdot_fit']        = projdot_fit
fit_results['projdot_fit_params'] = projdot_fit_params    # columns: A, omega_dom (interp), phi
fit_results['projdot_rel_err']    = projdot_rel_err

fit_results['fft_peak_freqs']     = fft_peak_freqs_interp  # interpolated (primary)
fit_results['fft_peak_freqs_bin'] = fft_peak_freqs_bin     # bin-center (reference)
fit_results['fft_peak_powers']    = fft_peak_powers
fit_results['fft_peak_amps']      = fft_peak_amps
fit_results['fft_peak_phases']    = fft_peak_phases

out_path = output_prefix + '_' + output_suffix + '.npy'
print(f"\nSaving output as: {out_path}")
np.save(out_path, fit_results)

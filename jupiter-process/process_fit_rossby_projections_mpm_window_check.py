"""
Test sensitivity of MPM frequency estimates to window length, for each mode.
Used to diagnose whether frequency splitting is a windowing artifact or a
property of the signal itself.

Usage:
    test_mpm_window_sensitivity.py <processed_file> [options]

Options:
    --K=<int>               number of sinusoidal components to fit per mode [default: 8]
    --sv_thresh=<float>     singular value threshold for MPM [default: 5e-1]
    --fracs=<str>           comma-separated window fractions to test [default: 1.0,0.9,0.8,0.7,0.6]
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from docopt import docopt

### read in options ###
args = docopt(__doc__)
print(args)

processed_file_str = args['<processed_file>']
K = int(args['--K'])
sv_thresh_opt = float(args['--sv_thresh'])
fracs = [float(f) for f in args['--fracs'].split(',')]

### fitting functions ###
def fit_multicomp(t, y, omegas):
    cols = []
    for omega in omegas:
        cols += [np.sin(omega * t), np.cos(omega * t)]
    A_mat = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)

    amps, phases, fits = [], [], []
    for k, omega in enumerate(omegas):
        c1, c2 = coeffs[2*k], coeffs[2*k + 1]
        amps.append(np.sqrt(c1**2 + c2**2))
        phases.append(np.arctan2(c2, c1))
        fits.append(A_mat[:, 2*k:2*k + 2] @ coeffs[2*k: 2*k + 2])

    amp_order = np.argsort(amps)[::-1]
    return (np.array(amps)[amp_order],
            np.array(phases)[amp_order],
            omegas[amp_order],
            np.array(fits)[amp_order],
            A_mat @ coeffs)

def matrix_pencil_frequency(y, dt, K_max=8, sv_thresh=sv_thresh_opt):
    """
    Estimate K frequencies via MPM (Matrix Pencil Method)
    """
    N = len(y)
    L = N // 2

    Y = np.array([y[i:i + L + 1] for i in range(N - L)])

    U, s, Vh = np.linalg.svd(Y, full_matrices=False)

    s_norm = s / s[0]
    K_thresh = int(np.sum(s_norm > sv_thresh))
    if K_thresh < 2:
        print('  svd did not yield any values larger than specified threshold')
    K_thresh = np.max((2, K_thresh))
    if K_thresh % 2 != 0:
        K_thresh += 1
    K_used = np.min((K_thresh, K_max))

    U1 = U[:, :2*K_used]
    U1_top = U1[:-1, :]
    U1_bot = U1[1:, :]

    Z = np.linalg.pinv(U1_top) @ U1_bot
    poles = np.linalg.eigvals(Z)
    log_poles = np.log(poles) / dt
    freqs = log_poles.imag
    decays = log_poles.real

    pos_mask = freqs > 0
    if np.sum(pos_mask) == 0:
        print('  warning: no positive parts found', freqs)
    return freqs[pos_mask], decays[pos_mask], s_norm, K_used

### load processed projection file ###
logger.info("loading: " + processed_file_str)
processed = np.load(processed_file_str, allow_pickle=True)[()]

tw           = processed['tw']
idxs_include = processed['idxs_include']
projl2_track = processed['projl2']
evals_re     = np.copy(processed['evals_re'])
evals_im     = np.copy(processed['evals_im'])
drifts       = np.copy(processed['drifts'])

nidxs = len(idxs_include)
nw    = len(tw)
dt    = tw[1] - tw[0]
T     = tw[-1] - tw[0]  # full window length

print('nw=%d, dt=%.4f, T=%.4f' % (nw, dt, T))
print('fracs to test:', fracs)
print('corresponding window lengths:', ['%.4f' % (f * T) for f in fracs])
print('corresponding spurious freq scale 2pi/T_frac:', ['%.4f' % (2*np.pi / (f * T)) for f in fracs])
print()

### first pass: run MPM on full window for all modes (mirrors main script) ###
print('=== first pass: full window MPM for all modes ===')
all_omegas  = [None] * nidxs
all_decays  = [None] * nidxs
all_s_norms = [None] * nidxs
all_amps    = [None] * nidxs

for i, idx in enumerate(idxs_include):
    print('idx=%d, eigenvalue=%.4f + i%.4f' % (idx, evals_re[idx], evals_im[idx]))
    signal = projl2_track[i, :]
    freqs, decays, s_norm, K_used = matrix_pencil_frequency(signal, dt, K_max=K)

    if len(freqs) == 0:
        print('  i=%d: MPM returned no positive frequencies' % i)
        continue

    amps, phases, omegas, component_fits, total_fit = fit_multicomp(tw, signal, freqs)

    all_omegas[i]  = omegas   # sorted by amplitude, greatest to least
    all_decays[i]  = decays
    all_s_norms[i] = s_norm[:2*K_used]
    all_amps[i]    = amps

    print('  K_used=%d, freqs (amp-sorted)=%s' % (K_used, np.round(omegas, 4)))
    print('  amps=%s' % np.round(amps, 6))
    print('  s_norm (retained): %s' % np.round(s_norm[:2*K_used], 4))

### identify dominant mode ###
dom_i     = np.argmax([amps[0] if amps is not None else 0. for amps in all_amps])
omega_dom = all_omegas[dom_i][0]
print()
print('dominant mode: i=%d, omega_dom=%.4f' % (dom_i, omega_dom))

### window sensitivity test ###
print()
print('=== window sensitivity test ===')
for i, idx in enumerate(idxs_include):
    if all_omegas[i] is None:
        continue

    signal      = projl2_track[i, :]
    omega_lt    = evals_re[idx]

    print()
    print('--- i=%d, idx=%d, linear theory omega=%.4f ---' % (i, idx, omega_lt))
    print('  %-6s  %-6s  %-8s  %-8s  %-10s  %s'
          % ('frac', 'n', 'T_frac', '2pi/T', 'K_used', 'freqs (positive)'))

    results = []  # store for splitting analysis below
    for frac in fracs:
        n       = int(frac * nw)
        t_frac  = n * dt
        signal_trunc = signal[:n]

        freqs, decays, s_norm, K_used = matrix_pencil_frequency(signal_trunc, dt, K_max=K)

        if len(freqs) == 0:
            print('  frac=%.2f: MPM returned no positive frequencies' % frac)
            results.append((frac, n, t_frac, K_used, np.array([])))
            continue

        freqs_sorted = np.sort(freqs)[::-1]
        results.append((frac, n, t_frac, K_used, freqs_sorted))

        print('  %-6.2f  %-6d  %-8.3f  %-8.4f  %-10d  %s'
              % (frac, n, t_frac, 2*np.pi/t_frac, K_used, np.round(freqs_sorted, 4)))

    # splitting analysis: for each pair of returned freqs, print their difference
    # and compare to 2pi/T_frac to test the windowing artifact hypothesis
    print()
    print('  splitting analysis (differences between adjacent freqs vs 2pi/T_frac):')
    print('  %-6s  %-8s  %-12s  %s'
          % ('frac', '2pi/T', 'freq diffs', 'ratio diff/(2pi/T)'))
    for frac, n, t_frac, K_used, freqs_sorted in results:
        if len(freqs_sorted) < 2:
            continue
        spurious_scale = 2 * np.pi / t_frac
        diffs = np.diff(freqs_sorted[::-1])  # differences between adjacent freqs, smallest first
        ratios = diffs / spurious_scale
        print('  %-6.2f  %-8.4f  %-12s  %s'
              % (frac, spurious_scale, np.round(diffs, 4), np.round(ratios, 3)))

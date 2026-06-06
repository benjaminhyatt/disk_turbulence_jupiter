"""
Summary figure of dominant Rossby mode FFT fitting results across gamma values.

Loads processed FFT fitting results for multiple simulations (different gamma)
and produces a two-panel summary figure:

  Panel 1: Dominant mode frequency (omega_selected) vs gamma, with:
           - Linear theory dispersion relation curves (dashed)
           - EVP eigenvalue predictions (small markers, one per mode per gamma)
           - Simulation selected frequency (large filled markers)

  Panel 2: Dominant mode amplitude (extrema-based) vs gamma.

Usage:
    plot_rossby_summary.py <fft_file>... [options]

Options:
    --output=<str>    output figure filename [default: rossby_summary.png]
    --mode_idx=<int>  which amplitude-sorted mode index to treat as dominant [default: 0]
    --n_evp=<int>     number of EVP eigenvalues to show per gamma [default: 3]
"""

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import re

### read in options ###
args = docopt(__doc__)
fft_files  = args['<fft_file>']
output     = args['--output']
mode_idx   = int(args['--mode_idx'])
n_evp      = int(args['--n_evp'])

### helper: extract gamma from filename ###
def extract_gamma(filepath):
    match = re.search(r'gam_([0-9a-zA-Z]+)', filepath)
    if match is None:
        raise ValueError(f"Could not extract gamma from filename: {filepath}")
    gam_str = match.group(1)

    m = re.match(r'(\d+)d(\d+)ep(\d+)', gam_str)
    if m:
        mantissa = float(m.group(1)) + float(m.group(2)) / 10
        exp      = int(m.group(3))
        return mantissa * 10**exp

    m = re.match(r'(\d+)ep(\d+)', gam_str)
    if m:
        mantissa = float(m.group(1))
        exp      = int(m.group(2))
        return mantissa * 10**exp

    try:
        return float(gam_str)
    except ValueError:
        raise ValueError(f"Could not parse gamma string: {gam_str}")

### load data ###
gamma_vals      = []
omega_selected  = []
omega_evp_dom   = []
amp_extrema     = []
evp_match_flags = []
evp_evals_all   = []   # list of arrays, shape (n_evp,) per gamma

for fpath in fft_files:
    try:
        gamma = extract_gamma(fpath)
    except ValueError as e:
        print(f"Warning: {e}, skipping")
        continue

    data = np.load(fpath, allow_pickle=True)[()]

    i = mode_idx

    sel_idx   = data['fft_selected_idx'][i]
    omega_sel = data['fft_peak_freqs_centroid'][i, sel_idx]
    evp_match = data['fft_evp_match'][i]
    omega_evp = np.abs(data['evals_re'][i])
    amp       = data['projdot_amp_extrema'][i]

    # collect first n_evp EVP eigenvalues (all modes, sorted by im part = im_inc order)
    evals_re_all = np.abs(data['evals_re'])   # already sorted by amplitude after processing
    # we want them sorted by eigenvalue real part for display
    evp_sorted = np.sort(evals_re_all)[::-1][:n_evp]

    gamma_vals.append(gamma)
    omega_selected.append(omega_sel)
    omega_evp_dom.append(omega_evp)
    amp_extrema.append(amp)
    evp_match_flags.append(evp_match)
    evp_evals_all.append(evp_sorted)

    print(f"gamma={gamma:.1f}: omega_sel={omega_sel:.4f}, omega_evp={omega_evp:.4f}, "
          f"amp={amp:.4f}, evp_match={evp_match}")
    print(f"  EVP eigenvalues: {evp_sorted}")

# sort by gamma
sort_idx        = np.argsort(gamma_vals)
gamma_vals      = np.array(gamma_vals)[sort_idx]
omega_selected  = np.array(omega_selected)[sort_idx]
omega_evp_dom   = np.array(omega_evp_dom)[sort_idx]
amp_extrema     = np.array(amp_extrema)[sort_idx]
evp_match_flags = np.array(evp_match_flags)[sort_idx]
evp_evals_all   = [evp_evals_all[k] for k in sort_idx]

### linear theory dispersion curves ###
k_vals   = np.array([3.8317, 7.0156, 10.1735])
k_labels = [r'$k_{1,1}=3.83$', r'$k_{1,2}=7.02$', r'$k_{1,3}=10.17$']
gamma_smooth = np.logspace(np.log10(max(30, gamma_vals.min()*0.5)),
                           np.log10(gamma_vals.max()*1.5), 200)

### figure ###
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

# ── Panel 1: frequency vs gamma ──────────────────────────────────────────────
ax = axes[0]

# linear theory curves
colors_lt = ['C0', 'C1', 'C2']
for k, label, c in zip(k_vals, k_labels, colors_lt):
    omega_lt = gamma_smooth / k**2
    ax.loglog(gamma_smooth, omega_lt, ls='--', lw=1.2, color=c, alpha=0.6,
              label=f'Lin. theory {label}')

# EVP eigenvalue markers (small, one column per gamma)
# plot as small gray diamonds, slightly offset horizontally for visibility
for gi, (gamma, evals) in enumerate(zip(gamma_vals, evp_evals_all)):
    label = 'EVP eigenvalues' if gi == 0 else None
    ax.scatter([gamma] * len(evals), evals,
               marker='D', s=18, color='gray', alpha=0.7,
               zorder=3, label=label)

# simulation selected frequencies
match_mask    = evp_match_flags
no_match_mask = ~evp_match_flags

if np.any(match_mask):
    ax.scatter(gamma_vals[match_mask], omega_selected[match_mask],
               color='k', marker='o', s=80, zorder=5,
               label='Simulation (EVP match)')
if np.any(no_match_mask):
    ax.scatter(gamma_vals[no_match_mask], omega_selected[no_match_mask],
               color='red', marker='x', s=100, zorder=5, linewidths=1.5,
               label='Simulation (EVP fallback)')

ax.plot(gamma_vals, omega_selected, color='k', lw=0.8, alpha=0.5)

ax.set_xlabel(r'$\gamma$', fontsize=12)
ax.set_ylabel(r'$\omega$ (rad/time)', fontsize=12)
ax.set_title('Dominant mode frequency vs $\\gamma$', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# ── Panel 2: amplitude vs gamma ──────────────────────────────────────────────
ax = axes[1]

if np.any(match_mask):
    ax.scatter(gamma_vals[match_mask], amp_extrema[match_mask],
               color='k', marker='o', s=80, zorder=5,
               label='Simulation (EVP match)')
if np.any(no_match_mask):
    ax.scatter(gamma_vals[no_match_mask], amp_extrema[no_match_mask],
               color='red', marker='x', s=100, zorder=5, linewidths=1.5,
               label='Simulation (EVP fallback)')

ax.plot(gamma_vals, amp_extrema, color='k', lw=0.8, alpha=0.5)

ax.plot(gamma_vals, 100*np.array(gamma_vals)**(-1), color = 'green', linestyle='dashed', label=r'$\propto \gamma^{-1}$')

ax.set_xlabel(r'$\gamma$', fontsize=12)
ax.set_ylabel('Projection amplitude (extrema estimate)', fontsize=11)
ax.set_title('Dominant mode amplitude vs $\\gamma$', fontsize=11)
ax.set_xscale('log')

ax.set_yscale('log')

ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Rossby wave dominant mode summary across $\\gamma$ sweep',
             fontsize=12)
fig.savefig(output, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {output}")

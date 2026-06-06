"""
Summary plot comparing the force balance equilibrium radius r_eq to the
observed CPC orbital radius r_CPC, across gamma values.

Loads the saved results dictionaries from process_force_balance_oom.py
and produces a summary figure.

Usage:
    plot_force_balance_summary.py <results_file>... [options]

Options:
    --output=<str>    output figure filename [default: force_balance_summary.png]
"""

import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
import re

### read options ###
args          = docopt(__doc__)
results_files = args['<results_file>']
output        = args['--output']

### helper: extract gamma from filename ###
def extract_gamma(filepath):
    match = re.search(r'gam_([0-9a-zA-Z]+)', filepath)
    if match is None:
        # fall back to gamma stored in the results dict
        return None
    gam_str = match.group(1)
    m = re.match(r'(\d+)d(\d+)ep(\d+)', gam_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))/10) * 10**int(m.group(3))
    m = re.match(r'(\d+)ep(\d+)', gam_str)
    if m:
        return float(m.group(1)) * 10**int(m.group(2))
    return None

### load all results ###
gamma_vals   = []
r_CPC_vals   = []
r_eq_vals    = []    # first equilibrium radius (innermost)
ratio_vals   = []
A_wave_vals  = []

for fpath in results_files:
    data = np.load(fpath, allow_pickle=True)[()]

    gamma  = float(data['gamma'])
    r_CPC  = float(data['r_CPC'])
    r_eq   = data['r_eq']
    ratio  = float(data['ratio_at_rCPC'])
    A_wave = float(data['A_wave'])

    # take the first (innermost) equilibrium radius if multiple exist
    r_eq_use = float(r_eq[0]) if len(r_eq) > 0 else np.nan

    gamma_vals.append(gamma)
    r_CPC_vals.append(r_CPC)
    r_eq_vals.append(r_eq_use)
    ratio_vals.append(ratio)
    A_wave_vals.append(A_wave)

    print(f"gamma={gamma:.0f}: r_CPC={r_CPC:.4f}, r_eq={r_eq_use:.4f}, "
          f"ratio@r_CPC={ratio:.3f}, A={A_wave:.4f}")

# sort by gamma
sort_idx    = np.argsort(gamma_vals)
gamma_vals  = np.array(gamma_vals)[sort_idx]
r_CPC_vals  = np.array(r_CPC_vals)[sort_idx]
r_eq_vals   = np.array(r_eq_vals)[sort_idx]
ratio_vals  = np.array(ratio_vals)[sort_idx]
A_wave_vals = np.array(A_wave_vals)[sort_idx]

### figure ###
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# panel 1: r_eq and r_CPC vs gamma
ax = axes[0]
ax.plot(gamma_vals, r_CPC_vals, color='k',  marker='o', ms=7, lw=1.2,
        label=r'$r_\mathrm{CPC}$ (tracking mean)')
ax.plot(gamma_vals, r_eq_vals,  color='C2', marker='s', ms=7, lw=1.2,
        label=r'$r_\mathrm{eq}$ (force balance)')

# connect corresponding pairs with thin gray lines
for g, r_c, r_e in zip(gamma_vals, r_CPC_vals, r_eq_vals):
    if not np.isnan(r_e):
        ax.plot([g, g], [r_c, r_e], color='gray', lw=0.8, ls=':')

ax.set_xlabel(r'$\gamma$', fontsize=12)
ax.set_ylabel(r'$r$', fontsize=12)
ax.set_title(r'Observed $r_\mathrm{CPC}$ vs predicted $r_\mathrm{eq}$', fontsize=11)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

# panel 2: ratio |dr(omega_RW)| / (gamma * r_CPC) vs gamma
ax = axes[1]
ax.plot(gamma_vals, ratio_vals, color='C3', marker='o', ms=7, lw=1.2)
ax.axhline(1.0, color='k', ls='--', lw=1.0, label='Exact balance')
ax.set_xlabel(r'$\gamma$', fontsize=12)
ax.set_ylabel(r'$|\partial_r\omega_\mathrm{RW}| \,/\, \gamma r_\mathrm{CPC}$',
              fontsize=11)
ax.set_title(r'Force balance ratio at $r_\mathrm{CPC}$', fontsize=11)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Force balance summary across $\\gamma$ sweep', fontsize=12)
fig.savefig(output, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {output}")

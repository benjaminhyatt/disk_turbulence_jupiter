"""
Plot CPC velocity profiles extracted by process_tracking.py.

Produces a 2x2 grid of panels:
  - Left column:  full signed profile (rho in [-rho_half_width, rho_half_width])
  - Right column: folded profile (|rho| in [0, rho_half_width])
  - Top row:    u_phi^CPC
  - Bottom row: u_r^CPC

Raw and RW-subtracted curves are overlaid in each panel where available.

Usage:
    plot_cpc_velocity_profile.py <file> [options]

Options:
    --output=<str>    prefix for output figure filename [default: processed_tracking]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)

file_str      = args['<file>']
output_prefix = args['--output']
output_suffix = file_str.split(output_prefix + '_')[1].split('.')[0].split('/')[0]

### load ###
processed = np.load(file_str, allow_pickle=True)[()]

rho_pts     = processed['rho_profile_pts']       # signed, shape (2*n-1,)
rho_abs_pts = processed['rho_abs_profile_pts']   # folded, shape (n,)
rho_hw      = float(processed['rho_half_width'])

uphi_mean      = processed['uphi_CPC_mean']
ur_mean        = processed['ur_CPC_mean']
uphi_fold_mean = processed['uphi_CPC_fold_mean']
ur_fold_mean   = processed['ur_CPC_fold_mean']
n_vel_profile  = int(processed['n_vel_profile'])
n_vel_skipped  = int(processed.get('n_vel_skipped', 0))

has_sub = 'uphi_CPC_sub_mean' in processed
if has_sub:
    uphi_sub_mean      = processed['uphi_CPC_sub_mean']
    ur_sub_mean        = processed['ur_CPC_sub_mean']
    uphi_sub_fold_mean = processed['uphi_CPC_sub_fold_mean']
    ur_sub_fold_mean   = processed['ur_CPC_sub_fold_mean']
    n_vel_sub          = int(processed['n_vel_sub'])

r_locs       = np.array(processed['r_locs'],       dtype=float)
glitch_flags = np.array(processed['glitch_flags'], dtype=bool)
r_CPC        = np.mean(r_locs[~glitch_flags])

print(f"Loaded: {n_vel_profile} frames averaged, {n_vel_skipped} skipped")
print(f"RW-subtracted available: {has_sub}" +
      (f" ({n_vel_sub} frames)" if has_sub else ""))
print(f"r_CPC (tracking mean, clean) = {r_CPC:.4f}")

### figure style ###
dpi = 300
plt.rcParams['font.family']      = 'serif'
plt.rcParams['font.serif']       = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm']      = 'serif'
plt.rcParams['font.size']        = 11
plt.rcParams['figure.dpi']       = dpi

# layout: 2 cols, 2 rows
t_mar, b_mar, l_mar, r_mar = 0.30, 0.55, 0.60, 0.15
h_pad, w_pad = 0.45, 0.60
golden_mean    = (np.sqrt(5) - 1.) / 2.
h_plot = 1.0
w_plot = h_plot / golden_mean
h_total = t_mar + h_plot + h_pad + h_plot + b_mar
w_total = l_mar + w_plot + w_pad + w_plot + r_mar
fig_width = 7.5
scale = fig_width / w_total

fig = plt.figure(figsize=(scale * w_total, scale * h_total))

def ax_rect(col, row):
    """Return [left, bottom, width, height] in figure fractions for (col, row) in 0-index."""
    l = (l_mar + col * (w_plot + w_pad)) / w_total
    b = 1. - (t_mar + (row + 1) * h_plot + row * h_pad) / h_total
    w = w_plot / w_total
    h = h_plot / h_total
    return [l, b, w, h]

ax_tl = fig.add_axes(ax_rect(0, 0))   # u_phi, signed
ax_tr = fig.add_axes(ax_rect(1, 0))   # u_phi, folded
ax_bl = fig.add_axes(ax_rect(0, 1))   # u_r,   signed
ax_br = fig.add_axes(ax_rect(1, 1))   # u_r,   folded

col_raw = 'C0'
col_sub = 'C3'

def decorate(ax, xlab, ylab, vline=True):
    ax.axhline(0., color='gray', lw=0.7, zorder=0)
    if vline:
        ax.axvline(0., color='k', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

# ── top-left: u_phi signed ────────────────────────────────────────────────────
decorate(ax_tl, r'$\rho = r - r_{\rm loc}$', r'$u_\phi^{\rm CPC}$')
ax_tl.plot(rho_pts, uphi_mean, color=col_raw, lw=1.8,
           label=r'raw')
if has_sub:
    ax_tl.plot(rho_pts, uphi_sub_mean, color=col_sub, lw=1.8, ls='--',
               label=r'RW sub.')
ax_tl.set_title(r'$u_\phi^{\rm CPC}$, signed $\rho$', fontsize=10)
ax_tl.legend(fontsize=9, loc='best')

# ── top-right: u_phi folded ───────────────────────────────────────────────────
decorate(ax_tr, r'$|\rho|$', r'$u_\phi^{\rm CPC}$', vline=False)
ax_tr.plot(rho_abs_pts, uphi_fold_mean, color=col_raw, lw=1.8,
           label=r'raw')
if has_sub:
    ax_tr.plot(rho_abs_pts, uphi_sub_fold_mean, color=col_sub, lw=1.8, ls='--',
               label=r'RW sub.')
ax_tr.set_title(r'$u_\phi^{\rm CPC}$, folded $|\rho|$', fontsize=10)
ax_tr.legend(fontsize=9, loc='best')

# ── bottom-left: u_r signed ───────────────────────────────────────────────────
decorate(ax_bl, r'$\rho = r - r_{\rm loc}$', r'$u_r^{\rm CPC}$')
ax_bl.plot(rho_pts, ur_mean, color=col_raw, lw=1.8,
           label=r'raw')
#if has_sub:
#    ax_bl.plot(rho_pts, ur_sub_mean, color=col_sub, lw=1.8, ls='--',
#               label=r'RW sub.')
ax_bl.set_title(r'$u_r^{\rm CPC}$, signed $\rho$', fontsize=10)
ax_bl.legend(fontsize=9, loc='best')

# ── bottom-right: u_r folded ──────────────────────────────────────────────────
decorate(ax_br, r'$|\rho|$', r'$u_r^{\rm CPC}$', vline=False)
ax_br.plot(rho_abs_pts, ur_fold_mean, color=col_raw, lw=1.8,
           label=r'raw')
#if has_sub:
#    ax_br.plot(rho_abs_pts, ur_sub_fold_mean, color=col_sub, lw=1.8, ls='--',
#               label=r'RW sub.')
ax_br.set_title(r'$u_r^{\rm CPC}$, folded $|\rho|$', fontsize=10)
ax_br.legend(fontsize=9, loc='best')

# shared super-title
fig.text(0.5, 1. - 0.4*t_mar/h_total,
         f'CPC velocity profiles  '
         f'($N_{{\\rm frames}}={n_vel_profile}$, '
         f'$N_{{\\rm skip}}={n_vel_skipped}$, '
         f'$\\bar{{r}}_{{\\rm CPC}}={r_CPC:.3f}$)',
         ha='center', va='top', fontsize=10)

savepath = f'cpc_velocity_profile_{output_suffix}.pdf'
fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
plt.close(fig)
print(f'Figure saved: {savepath}')

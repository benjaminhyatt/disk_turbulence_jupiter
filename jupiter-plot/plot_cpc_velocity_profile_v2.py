"""
Plot CPC velocity profiles extracted by process_tracking.py.

Produces two panels per velocity component (u_phi^CPC and u_r^CPC):
  - Raw profile (lab frame velocity transformed to CPC frame)
  - RW-subtracted profile (if available in the file)

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

r_profile_pts = processed['r_profile_pts']
uphi_CPC_mean = processed['uphi_CPC_mean']
ur_CPC_mean   = processed['ur_CPC_mean']
n_vel_profile = int(processed['n_vel_profile'])

# RW-subtracted profiles (optional)
has_sub          = 'uphi_CPC_sub_mean' in processed
uphi_CPC_sub_mean = processed['uphi_CPC_sub_mean'] if has_sub else None
ur_CPC_sub_mean   = processed['ur_CPC_sub_mean']   if has_sub else None
n_vel_sub         = int(processed['n_vel_sub'])     if has_sub else 0

# r_CPC from clean tracking mean
r_locs       = np.array(processed['r_locs'], dtype=float)
glitch_flags = np.array(processed['glitch_flags'], dtype=bool)
r_CPC        = np.mean(r_locs[~glitch_flags])

print(f"Loaded velocity profile averaged over {n_vel_profile} frames")
print(f"RW-subtracted profile available: {has_sub}" +
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

t_mar, b_mar, l_mar, r_mar = (0.1, 0.2, 0.35, 0.1)
golden_mean    = (np.sqrt(5) - 1.) / 2.
h_plot, w_plot = (1., 1. / golden_mean)
h_pad          = 0.25 * h_plot
h_total        = t_mar + h_plot + h_pad + h_plot + b_mar
w_total        = l_mar + w_plot + r_mar
fig_width      = 5.5
scale          = fig_width / w_total

fig = plt.figure(figsize=(scale * w_total, scale * h_total))

left   = l_mar / w_total
width  = w_plot / w_total
height = h_plot / h_total

bottom1 = 1 - (t_mar + h_plot) / h_total
bottom2 = 1 - (t_mar + h_plot + h_pad + h_plot) / h_total

### panel 1: u_phi^CPC ###
ax1 = fig.add_axes([left, bottom1, width, height])
ax1.axhline(0., color='gray', lw=0.8)
ax1.axvline(r_CPC, color='k', ls='--', lw=1.0,
            label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
ax1.plot(r_profile_pts, uphi_CPC_mean, color='C0', lw=2.0,
         label=r'$\langle u_\phi^{\rm CPC} \rangle_t$ (raw)')
if has_sub:
    ax1.plot(r_profile_pts, uphi_CPC_sub_mean, color='C0', lw=2.0, ls='--',
             label=r'$\langle u_\phi^{\rm CPC} \rangle_t$ (RW sub.)')
ax1.set_xlabel(r'$r$')
ax1.set_ylabel(r'$u_\phi^{\rm CPC}$')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_title(f'CPC velocity profile  ($N_{{\\rm frames}}={n_vel_profile}$)',
              fontsize=10)

### panel 2: u_r^CPC ###
ax2 = fig.add_axes([left, bottom2, width, height])
ax2.axhline(0., color='gray', lw=0.8)
ax2.axvline(r_CPC, color='k', ls='--', lw=1.0,
            label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
ax2.plot(r_profile_pts, ur_CPC_mean, color='C1', lw=2.0,
         label=r'$\langle u_r^{\rm CPC} \rangle_t$ (raw)')
if has_sub:
    ax2.plot(r_profile_pts, ur_CPC_sub_mean, color='C1', lw=2.0, ls='--',
             label=r'$\langle u_r^{\rm CPC} \rangle_t$ (RW sub.)')
ax2.set_xlabel(r'$r$')
ax2.set_ylabel(r'$u_r^{\rm CPC}$')
ax2.legend(loc='lower right', fontsize=9)

savepath = f'cpc_velocity_profile_{output_suffix}.pdf'
fig.savefig(savepath, dpi=dpi)
plt.close(fig)
print(f'Figure saved: {savepath}')

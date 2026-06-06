"""
Compute the time-averaged, azimuthally-averaged zonal-flow profile U_zonal(r)
and its radial gradient dU_zonal/dr from an HDF5 analysis file.

This is the "Stage 1" naive estimate: U_zonal(r) is just the time + azimuthal
mean of the lab-frame u_phi component, with no attempt to subtract the CPC's
own m=0 projection. Near r = r_CPC, U_zonal(r) is contaminated by the CPC; at
radii outside the CPC's spatial footprint (~ rho_max from r_CPC) the
contamination is negligible and U_zonal(r) is a clean estimate of the
background zonal flow.

The gradient is computed via Dedalus on the disk basis so it's consistent with
how downstream scripts (e.g. process_force_balance_oom_rsweep_v6.py) evaluate
radial derivatives.

Usage:
    process_zonal_flow_v1.py <hdf5_file> [options]

Arguments:
    <hdf5_file>             HDF5 analysis file from Dedalus simulation

Options:
    --tracking_file=<str>   optional processed tracking .npy; if provided, the
                            plot will overlay r_CPC and a CPC-footprint band
                            [default: None]
    --t_start=<float>       sim time to begin averaging [default: 149.]
    --t_end=<float>         sim time to stop averaging  [default: 251.]
    --rho_window=<float>    half-width of the CPC-footprint band to shade in
                            the plot, around r_CPC [default: 0.1]
    --output=<str>          output .npy path; 'auto' to use the output_suffix
                            convention [default: auto]
    --plot=<str>            plot path; 'auto' to use the output_suffix
                            convention. Empty string or 'none' to suppress.
                            [default: auto]
    --output_prefix=<str>   prefix used when --output / --plot are 'auto'
                            [default: processed_zonal_flow]
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read options ###
args = docopt(__doc__)
print(args)

file_str       = args['<hdf5_file>']
tracking_arg   = args['--tracking_file']
t_start        = float(args['--t_start'])
t_end          = float(args['--t_end'])
rho_window     = float(args['--rho_window'])
output_arg     = args['--output']
plot_arg       = args['--plot']
output_prefix  = args['--output_prefix']
dealias        = 3/2

### filename parsing (same convention as the rest of the pipeline) ###
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]
Nphi       = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr         = int(output_suffix.split('Nr_')[1].split('_')[0])
alpha_read = str_to_float(output_suffix.split('alpha_')[1].split('_')[0])
gamma_read = str_to_float(output_suffix.split('gam_')[1].split('_')[0])
eps_read   = str_to_float(output_suffix.split('eps_')[1].split('_')[0])
nu_read    = str_to_float(output_suffix.split('nu_')[1].split('_')[0])

alpha_vals = np.array((2e-3, 1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 920, 950, 1200, 1920,
                       2372, 2500, 3200))
eps_vals   = np.array([3.3e-1, 1.0, 2.0])
nu_vals    = np.array([5e-5, 8/90000, 2e-4])
alpha = float(alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))])
gamma = float(gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))])
eps   = float(eps_vals[np.argmin(np.abs(eps_vals - eps_read))])
nu    = float(nu_vals[np.argmin(np.abs(nu_vals - nu_read))])

print(f"Parsed parameters: Nphi={Nphi}, Nr={Nr}, "
      f"alpha={alpha}, gamma={gamma}, eps={eps}, nu={nu}")

### resolve output paths ###
def auto_or(arg, default_template):
    if arg.lower() == 'auto':
        return default_template
    return arg

output_path = auto_or(output_arg, f"{output_prefix}_{output_suffix}.npy")
if plot_arg.lower() in ('none', '',):
    plot_path = None
else:
    plot_path = auto_or(plot_arg, f"{output_prefix}_{output_suffix}.png")
print(f"output .npy path: {output_path}")
print(f"plot path:        {plot_path}")

### Dedalus setup ###
coords     = d3.PolarCoordinates('phi', 'r')
dist       = d3.Distributor(coords, dtype=np.float64)
disk       = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                          dealias=dealias, dtype=np.float64)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
phi_1d = phi_deal[:, 0]
r_1d   = r_deal[0, :]
Nphi_deal, Nr_deal = len(phi_1d), len(r_1d)

u_field = dist.VectorField(coords, name='u', bases=disk)

er = dist.VectorField(coords, bases=disk)
er.change_scales(dealias)
er['g'][1] = 1.0

### open HDF5 and select frames ###
logger.info("Opening HDF5: " + file_str)
f = h5py.File(file_str, 'r')

if 'u' not in f['tasks']:
    raise RuntimeError("HDF5 file does not contain a 'u' velocity task.")

t_all = f['tasks/u'].dims[0]['sim_time'][:]
t_end_eff = min(t_end, float(t_all[-1]))
if t_start > t_all[-1]:
    raise ValueError(f"t_start={t_start} beyond last available time {t_all[-1]:.3f}")

ws_start = np.where(t_all <= t_start)[0][-1] if np.any(t_all <= t_start) else 0
ws_end   = np.where(t_all >= t_end_eff)[0][0]
ws       = np.arange(ws_start, ws_end + 1)
nw, tw   = len(ws), t_all[ws]
print(f"Processing {nw} writes: t={tw[0]:.3f} to t={tw[-1]:.3f}")

### accumulate the azimuthal mean of u_phi over time ###
u_phi_m0_sum         = np.zeros(Nr_deal)
u_phi_m0_time_series = np.zeros((nw, Nr_deal))

prog_cad = max(1, nw // 50)
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print(f"writes loop: i={i} out of {nw}")
    u_field.load_from_hdf5(f, w)
    u_field.change_scales(dealias)
    uphi_g = u_field['g'][0]                # shape (Nphi_deal, Nr_deal)
    u_phi_m0 = np.mean(uphi_g, axis=0)      # shape (Nr_deal,)
    u_phi_m0_time_series[i, :] = u_phi_m0
    u_phi_m0_sum += u_phi_m0

f.close()
U_zonal = u_phi_m0_sum / nw                 # time-averaged

### compute radial gradient via Dedalus ###
# build a 2D field on the disk that is constant in phi, then take d/dr.
U_zonal_2d = dist.Field(bases=disk)
U_zonal_2d.change_scales(dealias)
U_zonal_2d['g'] = np.broadcast_to(U_zonal[np.newaxis, :], (Nphi_deal, Nr_deal)).copy()

dU_dr_2d = (er @ d3.grad(U_zonal_2d)).evaluate()
dU_dr_2d.change_scales(dealias)
dU_zonal_dr = dU_dr_2d['g'][0, :]           # same for every phi by construction

### optional: load tracking file to get r_CPC for overlays ###
r_CPC = None
if tracking_arg not in ('None', 'none', None):
    logger.info("Loading tracking file: " + tracking_arg)
    tracking = np.load(tracking_arg, allow_pickle=True)[()]
    r_locs       = np.array(tracking['r_locs'],   dtype=float)
    glitch_flags = np.array(tracking.get('glitch_flags',
                                         np.zeros_like(r_locs)), dtype=bool)
    r_clean      = r_locs[~glitch_flags]
    if len(r_clean) > 0:
        r_CPC = float(np.mean(r_clean))
        print(f"r_CPC from tracking file: {r_CPC:.4f}")

### diagnostic prints ###
# rough scale comparison: |dU/dr| at a few representative radii vs gamma*r
print(f"\n{'r':>8}  {'U(r)':>10}  {'dU/dr':>12}  {'gamma*r':>12}  {'|dU/dr|/(gamma*r)':>20}")
print('-' * 70)
step = max(1, Nr_deal // 30)
for ri in range(0, Nr_deal, step):
    rval = r_1d[ri]
    if rval < 1e-3:
        continue
    ratio = abs(dU_zonal_dr[ri]) / (gamma * rval + 1e-30)
    marker = ''
    if r_CPC is not None and abs(rval - r_CPC) < 2 * (r_1d[1] - r_1d[0]):
        marker = '  <-- r_CPC'
    print(f"{rval:>8.4f}  {U_zonal[ri]:>10.4f}  {dU_zonal_dr[ri]:>12.4e}  "
          f"{gamma * rval:>12.4e}  {ratio:>20.4f}{marker}")

### save ###
processed = {
    'r_1d'                  : r_1d,
    'phi_1d'                : phi_1d,
    'U_zonal'               : U_zonal,
    'dU_zonal_dr'           : dU_zonal_dr,
    'u_phi_m0_time_series'  : u_phi_m0_time_series,
    'tw'                    : tw,
    'gamma'                 : gamma,
    'alpha'                 : alpha,
    'eps'                   : eps,
    'nu'                    : nu,
    'Nphi'                  : Nphi,
    'Nr'                    : Nr,
    'output_suffix'         : output_suffix,
    'r_CPC'                 : r_CPC,
    'rho_window'            : rho_window,
}
np.save(output_path, processed)
print(f"\nSaved: {output_path}")

### plot ###
if plot_path is not None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True,
                              sharex=True)

    # panel 1: U_zonal(r)
    ax = axes[0]
    ax.plot(r_1d, U_zonal, color='C0', lw=1.5, label=r'$U_{\rm zonal}(r)$')
    ax.axhline(0, color='gray', lw=0.5)
    if r_CPC is not None:
        ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
                   label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
        ax.axvspan(max(0, r_CPC - rho_window), r_CPC + rho_window,
                   color='gray', alpha=0.15,
                   label=f'CPC footprint $\\pm{rho_window:.2f}$')
    ax.set_ylabel(r'$U_{\rm zonal}(r) \;=\; \langle u_\phi\rangle_{\phi,t}$')
    ax.set_title(f'Naive zonal flow profile — $\\gamma={gamma:.0f}$, '
                 f'$t\\in[{tw[0]:.1f}, {tw[-1]:.1f}]$, {nw} frames')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # panel 2: dU/dr and gamma*r
    ax = axes[1]
    ax.plot(r_1d, dU_zonal_dr, color='C1', lw=1.5,
            label=r'$\partial_r U_{\rm zonal}$')
    ax.plot(r_1d,  gamma * r_1d, color='C0', lw=1.0, ls=':',
            label=r'$+\gamma r$')
    ax.plot(r_1d, -gamma * r_1d, color='C0', lw=1.0, ls=':',
            label=r'$-\gamma r$')
    ax.axhline(0, color='gray', lw=0.5)
    if r_CPC is not None:
        ax.axvline(r_CPC, color='k', ls='--', lw=1.0,
                   label=f'$r_{{\\rm CPC}}={r_CPC:.4f}$')
        ax.axvspan(max(0, r_CPC - rho_window), r_CPC + rho_window,
                   color='gray', alpha=0.15)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'radial gradient')
    ax.set_title('Zonal-flow gradient vs the gamma forcing scale')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

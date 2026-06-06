"""
plot_spectral_budget.py

2x3 multi-panel visualization of the spectral energy budget for 2D disk
turbulence.  Loads 1-3 processed .npy files produced by:

    process_spectra_optionA.py        (H=+1 Dini, scalar J_m basis)
    process_spectra_zb_mbin_v3.py     (Route C: Dir/Robin, J_{|m-1|})
    process_spectra_optionG.py        (streamfunction psi, scalar Dirichlet)

Auto-detects which option each file came from by inspecting keys.

Panels (2 rows x 3 cols):
    (0, 0) E(k, t)        log-log, positive
    (0, 1) T(k, t)        linear-y log-x, signed (zero line shown)
    (0, 2) dE/dt(k, t)    linear-y log-x, signed
    (1, 0) D_nu(k, t)     log-log, positive
    (1, 1) D_alpha(k, t)  log-log, positive
    (1, 2) Budget residual = dE/dt - T + D_nu + D_alpha
                          linear-y log-x, signed; ~ -F (so ~ -eps in forcing
                          band, ~0 elsewhere)

Each panel: current-frame curve (bold).  In --frames and --keyframes modes,
the time-averaged steady-state curve is shown faintly in the background.
When multiple input files are passed, each panel overlays one line per
option, distinguished by colour + linestyle.

Modes (mutually exclusive):
  --steady        Default. One PDF, time-averaged-steady-state curves only.
  --frames        One PNG per saved snapshot. Output goes to
                  ./frames_<suffix>/budget_frame_NNNN.png .
  --keyframes=<times>
                  Comma-separated times (e.g. "0.05,0.5,2.0,10").  One PDF
                  per requested time; nearest-saved-snapshot is used.

Optional flags:
  --kmax_truncate    Cap the x-axis at k ~ bin_centers[Nr/2] (the Nyquist
                     limit on the radial Bessel grid).  Default off.
  --scaling_lines    Overlay k^(-5/3) and k^(-3) reference slopes on the
                     E(k) panel.  Default off.
  --outdir=<path>    Output directory [default: .].

Usage:
    plot_spectral_budget.py <file>... [options]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

from docopt import docopt
args = docopt(__doc__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

mpl.rcParams['figure.dpi']    = 100
mpl.rcParams['savefig.dpi']   = 200
mpl.rcParams['axes.grid']     = True
mpl.rcParams['grid.alpha']    = 0.3
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

OPTION_STYLES = {
    'A': dict(color='tab:blue',   ls='-',  label='A (H=+1 Dini, J_m)'),
    'C': dict(color='tab:orange', ls='--', label=r'C (Dir/Robin, $J_{|m-1|}$)'),
    'G': dict(color='tab:green',  ls=':',  label=r'G ($\psi$ Dirichlet)'),
}

# ---------------------------------------------------------------------------
# Option auto-detection
# ---------------------------------------------------------------------------

def detect_option(data):
    """Inspect keys to determine which post-processing option produced this
    .npy file.  Returns 'A', 'C', or 'G'.
    """
    keys = set(data.keys())
    # Option G has E_psiBn / T_psiBn naming.
    #if ('E_psiBn' in keys) or ('E_psiBn_tavg' in keys):
    if ('std_zs' in keys) or ('have_nonlin_omega' in keys):
        return 'G'
    # Option C (v3) has split keBn_r / keBn_p, and the H=-1 Robin lambda_R.
    if ('keBn_r' in keys) or ('keBn_r_tavg' in keys) or ('lambda_R' in keys):
        return 'C'
    # Option A has dini_zs and a top-level H key from the v2-style pipeline.
    if ('dini_zs' in keys) or ('H' in keys):
        return 'A'
    raise ValueError(
        "Cannot auto-detect option from .npy file: keys = "
        + ', '.join(sorted(map(str, keys))[:20]) + ' ...'
    )

# ---------------------------------------------------------------------------
# Canonical-name extraction of budget quantities
# ---------------------------------------------------------------------------

def _get(data, name, time_idx, fallback_compute=None):
    """Pull `name` (or `name+'_tavg'`) from data, indexing by time_idx as
    appropriate.  Returns a 1D array of length Nbins.  If `time_idx` is None,
    returns the time-averaged value (with fallback to a mean over the time
    axis if the _tavg version is absent)."""
    if time_idx is None:
        tavg_key = name + '_tavg'
        if tavg_key in data:
            return data[tavg_key]
        if name in data:
            arr = data[name]
            if arr.ndim >= 2:
                return arr.mean(axis=0)
            return arr
        if fallback_compute is not None:
            return fallback_compute()
        raise KeyError(f"Neither {tavg_key} nor {name} present in data")
    else:
        if name in data:
            return data[name][time_idx]
        if fallback_compute is not None:
            return fallback_compute()
        raise KeyError(f"{name} not present in data (per-snapshot mode)")


def extract_budget(data, option, time_idx=None):
    """Return canonical budget arrays {E, T, D_nu, D_alpha, dE_dt, residual,
    k} from a loaded .npy dict, picking the right keys for the option.
    `time_idx=None` → time-averaged steady-state values.  Otherwise pulls
    the snapshot at that index.
    """
    k = data['bin_centers']
    
    if option in ('A', 'C', 'G'):
        E       = _get(data, 'keBn',       time_idx)
        T       = _get(data, 'tBn',        time_idx)
        D_nu    = _get(data, 'D_nu_Bn',    time_idx)
        D_alpha = _get(data, 'D_alpha_Bn', time_idx)
        dE_dt   = _get(data, 'dkeBn_dt',   time_idx)
    #elif option == 'G':
    #    E       = _get(data, 'E_psiBn',     time_idx)
    #    T       = _get(data, 'T_psiBn',     time_idx)
    #    D_nu    = _get(data, 'D_nu_Bn',     time_idx)
    #    D_alpha = _get(data, 'D_alpha_Bn',  time_idx)
    #    dE_dt   = _get(data, 'dE_psiBn_dt', time_idx)
    else:
        raise ValueError(f"Unknown option: {option}")

    # Budget residual: dE/dt - T + D_nu + D_alpha (≈ F where forced, ~0 else)
    residual_key = 'budget_residual_Bn'
    try:
        residual = _get(data, residual_key, time_idx)
    except KeyError:
        residual = dE_dt - T + D_nu + D_alpha

    return dict(E=E, T=T, D_nu=D_nu, D_alpha=D_alpha, dE_dt=dE_dt,
                residual=residual, k=k)

# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

PANEL_LAYOUT = [
    # (key,      title,                       yscale,   loc (row, col))
    ('E',        r'$E(k,\,t)$',               'log',    (0, 0)),
    ('T',        r'$T(k,\,t)$',               'linear', (0, 1)),
    ('dE_dt',    r'$\partial_t E(k,\,t)$',    'linear', (0, 2)),
    ('D_nu',     r'$D_\nu(k,\,t)$',           'log',    (1, 0)),
    ('D_alpha',  r'$D_\alpha(k,\,t)$',        'log',    (1, 1)),
    ('residual', r'$\partial_t E - T + D_\nu + D_\alpha\ (\approx -F)$',
                                              'linear', (1, 2)),
]


def _apply_kmax(ax, bdg, kmax_truncate):
    if kmax_truncate:
        Nr_local = len(bdg['k'])
        kmax = bdg['k'][Nr_local // 2 - 1]
        ax.set_xlim(right=kmax)


def _add_scaling_lines(ax, bdg):
    """k^(-5/3) and k^(-3) guide lines anchored at one-third into the k range.
    """
    k = bdg['k']
    E = bdg['E']
    # Pick a reference k near the middle-third of the inertial range.
    i_ref = max(1, len(k) // 4)
    k_ref = k[i_ref]
    E_ref = E[i_ref] if E[i_ref] > 0 else max(E.max(), 1e-30)

    k_53 = k[: max(2, len(k) // 2)]
    k_3  = k[len(k) // 5:]
    ax.plot(k_53, E_ref * (k_53 / k_ref) ** (-5/3),
            color='gray', ls='--', lw=0.8, alpha=0.5,
            label=r'$k^{-5/3}$')
    ax.plot(k_3,  E_ref * (k_3 / k_ref) ** (-3),
            color='gray', ls=':',  lw=0.8, alpha=0.5,
            label=r'$k^{-3}$')


def make_budget_figure(datasets, options, time_idx=None, t_value=None,
                       kmax_truncate=False, scaling_lines=False,
                       show_tavg_bg=False, title_extra=''):
    """Build the 2x3 budget figure.

    datasets, options: parallel lists.  Each dataset is the loaded .npy dict;
                       each option is the corresponding 'A'/'C'/'G' tag.
    time_idx:  snapshot index, or None for time-averaged-steady-state mode.
    t_value:   actual simulation time (for the title), or None.
    show_tavg_bg:  if True (use in --frames/--keyframes modes), plot the
                   time-averaged curves faintly in the background of each
                   panel under the current-frame curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), squeeze=False)

    # We'll grab one budget dict per (data, option) for layout-only purposes
    # (used by the kmax / scaling-lines helpers below); the actual plotted
    # data is read inside the panel loop.
    reference_bdg = extract_budget(datasets[0], options[0],
                                   time_idx=time_idx if time_idx is not None
                                            else None)

    for key, title, yscale, (row, col) in PANEL_LAYOUT:
        ax = axes[row, col]

        for data, opt in zip(datasets, options):
            style = OPTION_STYLES[opt]

            # Background: time-averaged steady-state (only in per-frame modes)
            if show_tavg_bg:
                bdg_tavg = extract_budget(data, opt, time_idx=None)
                y_bg = bdg_tavg[key]
                if yscale == 'log':
                    y_bg = np.where(y_bg > 0, y_bg, np.nan)
                ax.plot(bdg_tavg['k'], y_bg,
                        color=style['color'], ls=style['ls'],
                        alpha=0.25, lw=1.5)

            # Current frame (or steady curve in --steady mode)
            bdg = extract_budget(data, opt, time_idx=time_idx)
            y = bdg[key]
            if yscale == 'log':
                y = np.where(y > 0, y, np.nan)
            ax.plot(bdg['k'], y,
                    color=style['color'], ls=style['ls'],
                    lw=2.0, label=style['label'])

        ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.axhline(0, color='black', lw=0.5, alpha=0.4)

        _apply_kmax(ax, reference_bdg, kmax_truncate)

        if scaling_lines and key == 'E':
            _add_scaling_lines(ax, reference_bdg)

        ax.set_xlabel(r'$k$')
        ax.set_title(title)

    # Single legend on E panel
    axes[0, 0].legend(loc='best')

    # Suptitle
    if time_idx is None:
        suptitle = 'Time-averaged steady-state spectral budget'
    else:
        suptitle = (
            f'Spectral budget at t = {t_value:.4f}'
            if t_value is not None
            else f'Spectral budget at snapshot index {time_idx}'
        )
    if title_extra:
        suptitle += '   ' + title_extra
    fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def parse_input_suffix(filename):
    """Extract a meaningful suffix from an input filename like
    `processed_spectra_optionA_steady_<long_suffix>.npy` for use in output
    filenames.  Falls back to the stem of the file.
    """
    stem = Path(filename).stem  # drops .npy
    # Strip known prefixes
    for prefix in ('processed_spectra_optionA_steady_',
                   'processed_spectra_optionA_',
                   'processed_spectra_v3_steady_',
                   'processed_spectra_v3_',
                   'processed_spectra_optionG_steady_',
                   'processed_spectra_optionG_',
                   'processed_spectra_'):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def nearest_snapshot_index(ts, t_target):
    """Index in ts whose value is nearest t_target."""
    return int(np.argmin(np.abs(ts - t_target)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files  = args['<file>']
    outdir = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)

    kmax_truncate = bool(args.get('--kmax_truncate'))
    scaling_lines = bool(args.get('--scaling_lines'))

    # Determine mode
    if args.get('--frames'):
        mode = 'frames'
    elif args.get('--keyframes'):
        mode = 'keyframes'
        keyframe_times = [float(s) for s in args['--keyframes'].split(',')]
    else:
        mode = 'steady'

    # Load all files, auto-detect each
    datasets = []
    options  = []
    suffixes = []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)
        data = np.load(path, allow_pickle=True).item()
        opt  = detect_option(data)
        datasets.append(data)
        options.append(opt)
        suffixes.append(parse_input_suffix(f))
        print(f"Loaded {path.name}  ->  option {opt}")

    # Build a stem for output filenames.  If all input files share the same
    # base suffix, use it; otherwise concatenate option tags.
    unique_suffixes = list(dict.fromkeys(suffixes))
    if len(unique_suffixes) == 1:
        out_stem = unique_suffixes[0]
    else:
        out_stem = '_'.join(unique_suffixes)
    options_tag = ''.join(options)

    # Pull a time axis (from the first file).
    ts = np.atleast_1d(datasets[0]['ts'])

    if mode == 'steady':
        fig = make_budget_figure(
            datasets, options,
            time_idx=None,
            kmax_truncate=kmax_truncate,
            scaling_lines=scaling_lines,
            show_tavg_bg=False,
            title_extra=f"[options: {options_tag}]",
        )
        out = outdir / f"budget_steady_{options_tag}_{out_stem}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")

    elif mode == 'frames':
        framedir = outdir / f"frames_{options_tag}_{out_stem}"
        framedir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(ts):
            fig = make_budget_figure(
                datasets, options,
                time_idx=i, t_value=float(t),
                kmax_truncate=kmax_truncate,
                scaling_lines=scaling_lines,
                show_tavg_bg=True,
                title_extra=f"[options: {options_tag}]",
            )
            out = framedir / f"budget_frame_{i:04d}.png"
            fig.savefig(out)
            plt.close(fig)
            if i % 32 == 0:
                print(f"  wrote {out.name}")
        print(f"Wrote {len(ts)} frames to {framedir}")

    elif mode == 'keyframes':
        for t_req in keyframe_times:
            i = nearest_snapshot_index(ts, t_req)
            t_actual = float(ts[i])
            fig = make_budget_figure(
                datasets, options,
                time_idx=i, t_value=t_actual,
                kmax_truncate=kmax_truncate,
                scaling_lines=scaling_lines,
                show_tavg_bg=True,
                title_extra=f"[options: {options_tag}]",
            )
            out = outdir / f"budget_t{t_actual:.4f}_{options_tag}_{out_stem}.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(f"Wrote {out}  (requested t={t_req}, used nearest t={t_actual:.4f})")


if __name__ == '__main__':
    main()

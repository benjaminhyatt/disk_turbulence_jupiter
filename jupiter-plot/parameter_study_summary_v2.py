"""
parameter_study_summary.py

Aggregate diagnostics JSON files (one per parameter combination) produced
by summary_diagnostics.py, and produce a unified 1D trend visualisation
of how the key metrics depend on (alpha, nu, k_f).

The figure is organised around the primary question:
    *How do the fast and slow fractional losses vary with nu, alpha, k_f?*

Layout: 5 rows x 3 columns.  Each row is a metric, each column is one of
the three parameters as the abscissa.  Within each panel, lines are
colour-coded by one of the two remaining parameters and styled by the
other, so all (alpha, nu, k_f) combinations are visible at once.

  Row 0:  fast_loss_fraction (= A_fast / KE_initial)
  Row 1:  slow_loss_fraction (= A_slow / KE_initial)
  Row 2:  R (final retention) and R_after_fast overlaid
  Row 3:  r_fast and r_slow (with viscous nu*k_f^2 and frictional alpha
                              reference lines on the appropriate panel)
  Row 4:  sanity: eps_inf_total (should sit near 1) plus the f_nu/f_alpha
                  dissipation channel partition

Missing parameter combinations show up as line breaks (NaN points dropped).

Usage:
    parameter_study_summary.py <jsonfile>... [options]

Options:
    --outdir=<path>   Output directory [default: .]
    --method=<opt>    Which post-processing method to extract from each
                      JSON's 'methods' block [default: G]
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

from docopt import docopt
args = docopt(__doc__)

mpl.rcParams['figure.dpi']     = 100
mpl.rcParams['savefig.dpi']    = 200
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.alpha']     = 0.3
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9


# ---------------------------------------------------------------------------
# Reading & grouping
# ---------------------------------------------------------------------------

def load_records(files, method='G'):
    records = []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"WARN: {f} not found, skipping", file=sys.stderr)
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"WARN: failed to parse {f}: {e}", file=sys.stderr)
            continue
        params  = data.get('parameters', {})
        methods = data.get('methods', {})
        if method not in methods:
            available = list(methods.keys())
            if not available:
                print(f"WARN: {path.name} has no methods, skipping", file=sys.stderr)
                continue
            chosen = available[0]
            print(f"WARN: {path.name} does not contain method '{method}'; "
                  f"using '{chosen}' instead", file=sys.stderr)
            m = methods[chosen]
        else:
            m = methods[method]
        records.append({
            'file':  path.name,
            'alpha': params.get('alpha'),
            'nu':    params.get('nu'),
            'kf':    params.get('kf'),
            'eps':   params.get('eps'),
            **{k: v for k, v in m.items()},
        })
    return records


def unique_sorted_values(records, key):
    return sorted({r[key] for r in records if r.get(key) is not None})


# ---------------------------------------------------------------------------
# Panel-construction helper
# ---------------------------------------------------------------------------

LINESTYLES = ['-', '--', ':', '-.']


def _styles(records, color_param, style_param):
    """Return color_map (color_param value -> RGBA) and style_map (style_param
    value -> linestyle string)."""
    color_values = unique_sorted_values(records, color_param)
    style_values = unique_sorted_values(records, style_param)
    cmap = plt.cm.viridis
    if len(color_values) <= 1:
        color_map = {(color_values[0] if color_values else None): cmap(0.5)}
    else:
        color_map = {v: cmap(i / (len(color_values) - 1))
                     for i, v in enumerate(color_values)}
    style_map = {v: LINESTYLES[i % len(LINESTYLES)]
                 for i, v in enumerate(style_values)}
    return color_values, style_values, color_map, style_map


def _plot_one_metric(ax, records, metric_key, x_param,
                     color_param, style_param,
                     title, ylabel,
                     xscale='linear', yscale='linear',
                     show_legend=True):
    """Plot `metric_key` (y) vs `x_param` (x).  One line per
    (color_param, style_param) combination."""
    color_values, style_values, color_map, style_map = _styles(
        records, color_param, style_param)

    for cv in color_values:
        for sv in style_values:
            xs, ys = [], []
            for r in records:
                if r.get(color_param) != cv or r.get(style_param) != sv:
                    continue
                xv = r.get(x_param)
                yv = r.get(metric_key)
                if xv is None or yv is None:
                    continue
                if not (np.isfinite(xv) and np.isfinite(yv)):
                    continue
                xs.append(xv); ys.append(yv)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            ax.plot(xs, ys,
                    color=color_map[cv], ls=style_map[sv],
                    marker='o', markersize=4,
                    label=f'{color_param}={cv:g}, {style_param}={sv:g}')
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    ax.set_xlabel(x_param); ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='best', fontsize=6, ncol=1)


def _plot_two_metrics_overlay(ax, records, metric_keys, x_param,
                               color_param, style_param,
                               title, ylabel,
                               markers=None,
                               metric_labels=None,
                               xscale='linear', yscale='linear',
                               show_legend=True):
    """Overlay two metrics on the same axis, distinguished by marker shape
    and a per-metric alpha.  Used for (R, R_after_fast) overlay."""
    color_values, style_values, color_map, style_map = _styles(
        records, color_param, style_param)
    if markers is None:
        markers = ['o', '^']
    if metric_labels is None:
        metric_labels = metric_keys

    for k_metric, (mk, mark, mlbl) in enumerate(zip(metric_keys, markers, metric_labels)):
        for cv in color_values:
            for sv in style_values:
                xs, ys = [], []
                for r in records:
                    if r.get(color_param) != cv or r.get(style_param) != sv:
                        continue
                    xv = r.get(x_param); yv = r.get(mk)
                    if xv is None or yv is None:
                        continue
                    if not (np.isfinite(xv) and np.isfinite(yv)):
                        continue
                    xs.append(xv); ys.append(yv)
                if not xs:
                    continue
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                label = (f'{mlbl}: {color_param}={cv:g}, {style_param}={sv:g}'
                         if k_metric == 0 else
                         f'{mlbl}')
                ax.plot(xs, ys,
                        color=color_map[cv], ls=style_map[sv],
                        marker=mark, markersize=4,
                        alpha=1.0 if k_metric == 0 else 0.5,
                        label=label)
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    ax.set_xlabel(x_param); ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc='best', fontsize=6, ncol=1)


# ---------------------------------------------------------------------------
# Main trends figure (5 rows x 3 cols)
# ---------------------------------------------------------------------------

def make_trends_figure(records, eps_expected=1.0):
    fig, axes = plt.subplots(5, 3, figsize=(17.5, 22.0))

    # Column 0: vs alpha     -- colour by kf, linestyle by nu
    # Column 1: vs nu        -- colour by kf, linestyle by alpha
    # Column 2: vs kf        -- colour by alpha, linestyle by nu

    cols = [
        ('alpha', 'kf', 'nu',    'log', r'$\alpha$'),
        ('nu',    'kf', 'alpha', 'log', r'$\nu$'),
        ('kf',    'alpha', 'nu', 'linear', r'$k_f$'),
    ]

    # Row 0: fast_loss_fraction
    for j, (xp, cp, sp, xs, xlbl) in enumerate(cols):
        _plot_one_metric(
            axes[0, j], records, 'fast_loss_fraction', xp, cp, sp,
            title=rf'$A_{{\rm fast}}/KE_0$ vs {xlbl}', ylabel='fast loss frac',
            xscale=xs, yscale='linear',
            show_legend=(j == 2),
        )

    # Row 1: slow_loss_fraction
    for j, (xp, cp, sp, xs, xlbl) in enumerate(cols):
        _plot_one_metric(
            axes[1, j], records, 'slow_loss_fraction', xp, cp, sp,
            title=rf'$A_{{\rm slow}}/KE_0$ vs {xlbl}', ylabel='slow loss frac',
            xscale=xs, yscale='linear',
            show_legend=(j == 2),
        )

    # Row 2: R (final) and R_after_fast, overlaid
    for j, (xp, cp, sp, xs, xlbl) in enumerate(cols):
        _plot_two_metrics_overlay(
            axes[2, j], records, ['R', 'R_after_fast'], xp, cp, sp,
            title=rf'$R$ (filled), $R_{{\rm after\,fast}}$ (open) vs {xlbl}',
            ylabel='retention',
            markers=['o', '^'],
            metric_labels=['R', r'$R_{\rm after\,fast}$'],
            xscale=xs, yscale='linear',
            show_legend=(j == 2),
        )

    # Row 3: r_fast and r_slow.  Column 0: r_fast vs alpha.
    # Column 1: r_fast vs nu (with the nu * (k_f*pi)^2 reference line).
    # Column 2: r_slow vs alpha (with the r_slow = alpha reference line).
    _plot_one_metric(
        axes[3, 0], records, 'r_2exp_fast', 'alpha', 'kf', 'nu',
        title=r'$r_{\rm fast}$ vs $\alpha$', ylabel=r'$r_{\rm fast}$',
        xscale='log', yscale='log',
        show_legend=False,
    )
    _plot_one_metric(
        axes[3, 1], records, 'r_2exp_fast', 'nu', 'kf', 'alpha',
        title=r'$r_{\rm fast}$ vs $\nu$  (compare $\nu (k_f\pi)^2$)',
        ylabel=r'$r_{\rm fast}$',
        xscale='log', yscale='log',
        show_legend=True,
    )
    # nu * (k_f * pi)^2 reference per kf
    nus_uniq = unique_sorted_values(records, 'nu')
    kfs_uniq = unique_sorted_values(records, 'kf')
    cmap = plt.cm.viridis
    if nus_uniq and kfs_uniq:
        nu_arr = np.array(nus_uniq, dtype=float)
        for i, kf in enumerate(kfs_uniq):
            col = cmap(i / max(len(kfs_uniq) - 1, 1))
            axes[3, 1].plot(nu_arr, nu_arr * (kf * np.pi) ** 2,
                            ls=':', color=col, alpha=0.6,
                            label=rf'$\nu (k_f\pi)^2$, $k_f={kf:g}$')
        axes[3, 1].legend(loc='best', fontsize=6, ncol=1)

    _plot_one_metric(
        axes[3, 2], records, 'r_2exp_slow', 'alpha', 'kf', 'nu',
        title=r'$r_{\rm slow}$ vs $\alpha$  (compare $r=\alpha$)',
        ylabel=r'$r_{\rm slow}$',
        xscale='log', yscale='log',
        show_legend=True,
    )
    alpha_uniq = unique_sorted_values(records, 'alpha')
    if alpha_uniq:
        a_arr = np.array(alpha_uniq, dtype=float)
        axes[3, 2].plot(a_arr, a_arr, ls=':', color='gray',
                        label=r'$r_{\rm slow}=\alpha$')
        axes[3, 2].legend(loc='best', fontsize=6, ncol=1)

    # Row 4: sanity & dissipation partition
    for j, (xp, cp, sp, xs, xlbl) in enumerate(cols[:2]):
        _plot_one_metric(
            axes[4, j], records, 'res_total_ss', xp, cp, sp,
            title=rf'$\varepsilon_{{\rm inf}}$ (total) vs {xlbl}',
            ylabel=r'$\varepsilon_{\rm inf}$',
            xscale=xs, yscale='linear',
            show_legend=(j == 1),
        )
        axes[4, j].axhline(eps_expected, color='red', lw=0.8, alpha=0.6,
                            label=rf'$\varepsilon_{{\rm expected}}={eps_expected:g}$')

    # The last sanity panel: ratio f_nu / f_alpha (dissipation partition)
    # This tells us whether viscous dissipation dominates frictional or vice
    # versa over the full run, as a function of (alpha, nu, k_f).
    # Plot vs alpha, colour by kf, linestyle by nu.
    color_values, style_values, color_map, style_map = _styles(
        records, 'kf', 'nu')
    ax_partition = axes[4, 2]
    for cv in color_values:
        for sv in style_values:
            xs, ys = [], []
            for r in records:
                if r.get('kf') != cv or r.get('nu') != sv:
                    continue
                fn = r.get('f_nu'); fa = r.get('f_alpha'); a = r.get('alpha')
                if fn is None or fa is None or abs(fa) < 1e-30:
                    continue
                xs.append(a); ys.append(fn / fa)
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.array(xs)[order]; ys = np.array(ys)[order]
            ax_partition.plot(xs, ys,
                              color=color_map[cv], ls=style_map[sv],
                              marker='o', markersize=4,
                              label=f'kf={cv:g}, nu={sv:g}')
    ax_partition.set_xscale('log'); ax_partition.set_yscale('log')
    ax_partition.set_xlabel(r'$\alpha$')
    ax_partition.set_ylabel(r'$f_\nu / f_\alpha$')
    ax_partition.axhline(1.0, color='gray', lw=0.6, alpha=0.5)
    ax_partition.set_title(r'Dissipation partition $f_\nu / f_\alpha$ vs $\alpha$')
    ax_partition.legend(loc='best', fontsize=6, ncol=1)

    fig.suptitle('Parameter-study trends', y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Stdout summary table (unchanged from before)
# ---------------------------------------------------------------------------

def print_table(records, method, eps_expected=1.0):
    if not records:
        print('No records to summarise.')
        return
    print()
    print('=' * 110)
    print(f'Parameter-study summary  (method: {method})')
    print('=' * 110)
    hdr = f'{"alpha":>8}  {"nu":>10}  {"k_f":>6} | {"R":>7} {"R_fast":>7} ' \
          f'{"f_loss":>7} {"s_loss":>7} | {"eps":>7}  {"f_nu":>7} {"f_a":>7} | ' \
          f'{"r_fast":>8} {"r_slow":>8}'
    print(hdr)
    print('-' * len(hdr))
    rs = sorted(records, key=lambda r: (r.get('kf') or 0, r.get('alpha') or 0,
                                         r.get('nu') or 0))
    for r in rs:
        a  = r.get('alpha'); nu_ = r.get('nu'); kf = r.get('kf')
        R  = r.get('R',                  float('nan'))
        Rf = r.get('R_after_fast',       float('nan'))
        fl = r.get('fast_loss_fraction', float('nan'))
        sl = r.get('slow_loss_fraction', float('nan'))
        eps = r.get('res_total_ss',      float('nan'))
        fnu = r.get('f_nu',              float('nan'))
        fal = r.get('f_alpha',           float('nan'))
        rf  = r.get('r_2exp_fast',       float('nan'))
        rs_ = r.get('r_2exp_slow',       float('nan'))
        print(f'{a:>8g}  {nu_:>10g}  {kf:>6g} | '
              f'{R:>7.3g} {Rf:>7.3g} {fl:>7.3g} {sl:>7.3g} | '
              f'{eps:>7.3g} {fnu:>7.3g} {fal:>7.3g} | '
              f'{rf:>8.3g} {rs_:>8.3g}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files  = args['<jsonfile>']
    outdir = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    method = args['--method']

    records = load_records(files, method=method)
    if not records:
        print('No records loaded; nothing to do.', file=sys.stderr)
        sys.exit(1)

    alphas = unique_sorted_values(records, 'alpha')
    nus    = unique_sorted_values(records, 'nu')
    kfs    = unique_sorted_values(records, 'kf')
    eps_expected = float(records[0].get('eps') or 1.0)

    print(f'Loaded {len(records)} records.')
    print(f'  alphas: {alphas}')
    print(f'  nus:    {nus}')
    print(f'  k_fs:   {kfs}')

    fig = make_trends_figure(records, eps_expected=eps_expected)
    out = outdir / f'parameter_study_trends_{method}.pdf'
    fig.savefig(out)
    plt.close(fig)
    print(f'Wrote {out}')

    print_table(records, method, eps_expected=eps_expected)


if __name__ == '__main__':
    main()

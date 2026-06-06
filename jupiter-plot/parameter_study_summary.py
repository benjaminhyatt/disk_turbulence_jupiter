"""
parameter_study_summary.py

Aggregate diagnostics JSON files (one per parameter combination) produced
by summary_diagnostics.py, and visualise cross-parameter trends for the
2D disk turbulence study.

Reads any number of `diagnostics_*.json` files, groups by (alpha, nu, k_f),
selects a chosen method (default G), and produces:

(a) A grid of heatmaps over (alpha, nu), one column per k_f, one row per
    metric.  Metrics shown by default:
      - R         = KE_steady / KE_initial         (final retention)
      - R_after_fast                              (retention after the fast
                                                    transient)
      - fast_loss = A_fast / KE_initial            (energy lost during the
                                                    fast transient)
      - slow_loss = A_slow / KE_initial            (energy lost during the
                                                    slow approach to steady)
      - eps_total                                  (sanity check; should be
                                                    ~1 if budget closes)
      - f_nu / f_alpha                             (dissipation channel
                                                    partition ratio)

(b) Optional cross-parameter trend lines:
      - r_fast vs nu  (compare to nu * k_f^2 -- the viscous rate at the
                       forcing scale; line agreement indicates the fast
                       transient is genuinely viscous)
      - r_slow vs alpha (compare to alpha -- the friction rate; line
                         agreement indicates the slow approach is
                         genuinely frictional)
      - R vs alpha and R vs nu (separate panels, lines grouped by the
                                other parameter)

If a parameter combination is missing (run not yet finished, or JSON
missing), the corresponding heatmap cell shows up as a blank entry with
a labelled gap.

Usage:
    parameter_study_summary.py <jsonfile>... [options]

Options:
    --outdir=<path>   Output directory [default: .]
    --method=<opt>    Which post-processing method to extract from each
                      JSON's 'methods' block [default: G]
    --no_trends       Skip the cross-parameter trend-line figure.
    --no_heatmaps     Skip the heatmap figure.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import json
from pathlib import Path
import sys

from docopt import docopt
args = docopt(__doc__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

mpl.rcParams['figure.dpi']     = 100
mpl.rcParams['savefig.dpi']    = 200
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.alpha']     = 0.3
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9


# ---------------------------------------------------------------------------
# Reading & grouping
# ---------------------------------------------------------------------------

def load_records(files, method='G'):
    """Read each JSON file and pull out (alpha, nu, kf, metrics_dict) records
    for the requested method.  Returns a list of dicts."""
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
        params = data.get('parameters', {})
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
            'file': path.name,
            'alpha': params.get('alpha'),
            'nu':    params.get('nu'),
            'kf':    params.get('kf'),
            'eps':   params.get('eps'),
            **{k: v for k, v in m.items()},
        })
    return records


def unique_sorted_values(records, key):
    vals = sorted({r[key] for r in records if r.get(key) is not None})
    return vals


def build_matrix(records, kf, metric, alphas, nus):
    """Build an (n_alpha, n_nu) matrix of `metric` values at fixed kf."""
    out = np.full((len(alphas), len(nus)), np.nan)
    for r in records:
        if r['kf'] != kf:
            continue
        try:
            i = alphas.index(r['alpha'])
            j = nus.index(r['nu'])
        except ValueError:
            continue
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            out[i, j] = float(v)
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _heatmap(ax, M, alphas, nus, title, cmap='viridis',
             center=None, vmin=None, vmax=None, fmt='{:.3g}'):
    """Render M as a heatmap with annotated cells.  Rows: alpha (top-down by
    sorted ascending values).  Cols: nu."""
    if center is not None:
        finite = M[np.isfinite(M)]
        span = max(abs(finite.min() - center) if finite.size else 1.0,
                    abs(finite.max() - center) if finite.size else 1.0,
                    1e-12)
        norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
        im = ax.imshow(M, aspect='auto', cmap=cmap, norm=norm)
    else:
        im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(nus)));    ax.set_xticklabels([f'{x:g}' for x in nus], rotation=30)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f'{x:g}' for x in alphas])
    ax.set_xlabel(r'$\nu$');           ax.set_ylabel(r'$\alpha$')
    ax.set_title(title)
    # Cell annotations
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = fmt.format(v) if np.isfinite(v) else '—'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7,
                    color='white' if (np.isfinite(v) and (im.norm(v) > 0.5)) else 'black')
    plt.colorbar(im, ax=ax, shrink=0.85)


def make_heatmaps_figure(records, kfs, alphas, nus, eps_expected=1.0):
    metrics = [
        ('R',                  'R (final retention)',           'viridis', None,    None, None),
        ('R_after_fast',       'R after fast transient',         'viridis', None,    None, None),
        ('fast_loss_fraction', 'fast loss fraction (A_fast/KE_0)', 'plasma',  None,   None, None),
        ('slow_loss_fraction', 'slow loss fraction (A_slow/KE_0)', 'plasma',  None,   None, None),
        ('res_total_ss',       rf'$\varepsilon_{{\rm inf}}$ (total)',  'RdBu_r', float(eps_expected),  None, None),
        ('r_2exp_fast',        r'$r_{\rm fast}$ (2-exp)',        'magma',   None,    None, None),
        ('r_2exp_slow',        r'$r_{\rm slow}$ (2-exp)',        'magma',   None,    None, None),
    ]

    n_kf      = len(kfs)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_kf,
                              figsize=(4.0 * max(n_kf, 1) + 0.6,
                                       3.2 * n_metrics + 0.6),
                              squeeze=False)

    for i_m, (key, title, cmap, center, vmin, vmax) in enumerate(metrics):
        for i_k, kf in enumerate(kfs):
            ax = axes[i_m, i_k]
            M  = build_matrix(records, kf, key, alphas, nus)
            full_title = f'{title}\n$k_f = {kf:g}$' if i_m == 0 else title
            _heatmap(ax, M, alphas, nus, full_title, cmap=cmap, center=center,
                     vmin=vmin, vmax=vmax)

    fig.suptitle('Parameter-study summary — heatmaps over '
                 + rf'$(\alpha, \nu)$ at each $k_f$', y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    return fig


# ---------------------------------------------------------------------------
# Cross-parameter trend plots
# ---------------------------------------------------------------------------

def make_trends_figure(records, kfs, alphas, nus):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax_rfast_vs_nu  = axes[0, 0]
    ax_rslow_vs_a   = axes[0, 1]
    ax_R_vs_alpha   = axes[1, 0]
    ax_R_vs_nu      = axes[1, 1]

    cmap = plt.cm.viridis
    n_kf = max(len(kfs), 1)
    kf_colors = {kf: cmap(i / max(n_kf - 1, 1)) for i, kf in enumerate(kfs)}

    # r_fast vs nu, lines by (alpha, kf)
    for kf in kfs:
        for ialpha, alpha in enumerate(alphas):
            xs, ys = [], []
            for r in records:
                if r['kf'] != kf or r['alpha'] != alpha:
                    continue
                if r.get('nu') is None or not np.isfinite(r.get('r_2exp_fast', np.nan)):
                    continue
                xs.append(r['nu']); ys.append(r['r_2exp_fast'])
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                ax_rfast_vs_nu.plot(xs, ys,
                                    marker='o', ls='-',
                                    color=kf_colors[kf], alpha=0.4 + 0.6 * ialpha / max(len(alphas) - 1, 1),
                                    label=rf'$k_f={kf:g}$, $\alpha={alpha:g}$')
    # Reference line: r ~ nu * k_f^2 for each kf at one representative slope
    for kf in kfs:
        nus_arr = np.array(nus, dtype=float)
        # Scale line so it's visible: use the geometric mean nu as reference
        ref_nu = float(np.exp(np.mean(np.log(nus_arr))))
        ref_y  = ref_nu * (kf * np.pi) ** 2
        scale_factor = 1.0   # nominal
        ax_rfast_vs_nu.plot(nus_arr, scale_factor * nus_arr * (kf * np.pi) ** 2,
                            ls=':', color=kf_colors[kf], alpha=0.5,
                            label=rf'$\nu (k_f\pi)^2$ ref, $k_f={kf:g}$')
    ax_rfast_vs_nu.set_xscale('log'); ax_rfast_vs_nu.set_yscale('log')
    ax_rfast_vs_nu.set_xlabel(r'$\nu$'); ax_rfast_vs_nu.set_ylabel(r'$r_{\rm fast}$')
    ax_rfast_vs_nu.set_title(r'$r_{\rm fast}$ vs $\nu$ -- viscous-rate scaling check')
    ax_rfast_vs_nu.legend(loc='best', fontsize=7, ncol=2)

    # r_slow vs alpha
    for kf in kfs:
        for inu, nu in enumerate(nus):
            xs, ys = [], []
            for r in records:
                if r['kf'] != kf or r['nu'] != nu:
                    continue
                if r.get('alpha') is None or not np.isfinite(r.get('r_2exp_slow', np.nan)):
                    continue
                xs.append(r['alpha']); ys.append(r['r_2exp_slow'])
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                ax_rslow_vs_a.plot(xs, ys,
                                    marker='o', ls='-',
                                    color=kf_colors[kf], alpha=0.4 + 0.6 * inu / max(len(nus) - 1, 1),
                                    label=rf'$k_f={kf:g}$, $\nu={nu:g}$')
    # Reference line: r_slow = alpha
    alpha_arr = np.array(alphas, dtype=float)
    ax_rslow_vs_a.plot(alpha_arr, alpha_arr, ls=':', color='gray',
                       label=r'$r_{\rm slow}=\alpha$')
    ax_rslow_vs_a.set_xscale('log'); ax_rslow_vs_a.set_yscale('log')
    ax_rslow_vs_a.set_xlabel(r'$\alpha$'); ax_rslow_vs_a.set_ylabel(r'$r_{\rm slow}$')
    ax_rslow_vs_a.set_title(r'$r_{\rm slow}$ vs $\alpha$ -- frictional scaling check')
    ax_rslow_vs_a.legend(loc='best', fontsize=7, ncol=2)

    # R vs alpha (lines by nu, at each kf)
    for kf in kfs:
        for inu, nu in enumerate(nus):
            xs, ys = [], []
            for r in records:
                if r['kf'] != kf or r['nu'] != nu:
                    continue
                if r.get('alpha') is None or not np.isfinite(r.get('R', np.nan)):
                    continue
                xs.append(r['alpha']); ys.append(r['R'])
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                ax_R_vs_alpha.plot(xs, ys,
                                    marker='o', ls='-',
                                    color=kf_colors[kf], alpha=0.4 + 0.6 * inu / max(len(nus) - 1, 1),
                                    label=rf'$k_f={kf:g}$, $\nu={nu:g}$')
    ax_R_vs_alpha.set_xlabel(r'$\alpha$'); ax_R_vs_alpha.set_ylabel(r'$R$')
    ax_R_vs_alpha.set_xscale('log')
    ax_R_vs_alpha.set_title(r'$R$ (final retention) vs $\alpha$')
    ax_R_vs_alpha.legend(loc='best', fontsize=7, ncol=2)

    # R vs nu (lines by alpha, at each kf)
    for kf in kfs:
        for ialpha, alpha in enumerate(alphas):
            xs, ys = [], []
            for r in records:
                if r['kf'] != kf or r['alpha'] != alpha:
                    continue
                if r.get('nu') is None or not np.isfinite(r.get('R', np.nan)):
                    continue
                xs.append(r['nu']); ys.append(r['R'])
            if xs:
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                ax_R_vs_nu.plot(xs, ys,
                                marker='o', ls='-',
                                color=kf_colors[kf], alpha=0.4 + 0.6 * ialpha / max(len(alphas) - 1, 1),
                                label=rf'$k_f={kf:g}$, $\alpha={alpha:g}$')
    ax_R_vs_nu.set_xlabel(r'$\nu$'); ax_R_vs_nu.set_ylabel(r'$R$')
    ax_R_vs_nu.set_xscale('log')
    ax_R_vs_nu.set_title(r'$R$ (final retention) vs $\nu$')
    ax_R_vs_nu.legend(loc='best', fontsize=7, ncol=2)

    fig.suptitle('Parameter-study trends', y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Stdout summary table
# ---------------------------------------------------------------------------

def print_table(records, method, eps_expected=1.0):
    if not records:
        print('No records to summarise.')
        return
    print()
    print('=' * 96)
    print(f'Parameter-study summary  (method: {method})')
    print('=' * 96)
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
    files     = args['<jsonfile>']
    outdir    = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    method    = args['--method']
    no_trends = bool(args.get('--no_trends'))
    no_maps   = bool(args.get('--no_heatmaps'))

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

    if not no_maps:
        fig = make_heatmaps_figure(records, kfs, alphas, nus,
                                    eps_expected=eps_expected)
        out = outdir / f'parameter_study_heatmaps_{method}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'Wrote {out}')

    if not no_trends:
        fig = make_trends_figure(records, kfs, alphas, nus)
        out = outdir / f'parameter_study_trends_{method}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'Wrote {out}')

    print_table(records, method, eps_expected=eps_expected)


if __name__ == '__main__':
    main()

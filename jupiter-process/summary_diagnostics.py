"""
summary_diagnostics.py

Per-run summary diagnostics for the 2D disk turbulence parameter study.

Loads 1-3 processed .npy files (from process_spectra_optionA.py,
process_spectra_zb_mbin_v3.py [option C], or process_spectra_optionG.py)
and produces:

(a) A 2x2 multi-panel PDF figure showing:
    - KE_total(t) overlaid for all methods, with the disjoint 3-way band
      decomposition KE_low(t), KE_band(t), KE_high(t) as thin lines.
    - Cumulative dissipation channels cum_D_nu(t), cum_D_alpha(t), with
      KE_initial - KE_total(t) plotted as the reference (cumulative net
      energy lost).
    - Decay-rate diagnostic on log-y: KE_total(t) - KE_total_steady,
      with single- and two-exponential fits overlaid.
    - Per-band budget closure residual time series (forcing band only;
      the steady value of this equals the inferred epsilon).

(b) A small JSON of single-number summaries, one entry per loaded method:
    - KE_initial, KE_steady, retention R = KE_steady / KE_initial
    - Cumulative dissipation partition  f_nu = cum_D_nu(t_end) / KE_initial
                                        f_alpha = cum_D_alpha(t_end) / KE_initial
    - Inferred injection rate eps_inferred from band-integrated steady-state
      budget closure (should match the IVP's eps -- by default 1).
    - Per-band steady-state residuals (should be ~ -eps in band, ~0 elsewhere)
    - Single- and two-exponential decay rates plus AIC for model comparison

The JSON is named by the input file's parameter-set stem (auto-extracted)
so a downstream parameter_study_summary.py can collect across a parameter
study and produce cross-parameter aggregate plots.

Band partitioning (matching the IVP's forcing setup, with a half-bin pad
on each side so that the disk-basis spread of the Cartesian forcing shell
gets captured rather than split between adjacent bands):
    KE_low(t)  = integral_{k <  k_force * pi - pi/2} E(k, t) dk
    KE_band(t) = integral_{k_force * pi - pi/2 <= k < (k_force+1) * pi + pi/2} E(k, t) dk
    KE_high(t) = integral_{k >= (k_force+1) * pi + pi/2} E(k, t) dk
    Disjoint, sum to KE_total(t).
 
The padding can be disabled with --strict_cartesian_band, recovering the
narrow [k_f*pi, (k_f+1)*pi] convention.

Usage:
    summary_diagnostics.py <file>... [options]

Options:
    --outdir=<path>          Output directory [default: .]
    --t_steady_frac=<float>  Fraction of the time range to use for the
                             steady-state average (taken from the end)
                             [default: 0.5]
    --no_json                Skip the JSON output
    --no_figure              Skip the figure output
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

from scipy.optimize import curve_fit

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

# Distinguishable colour/linestyle per option, matching plot_spectral_budget.py
OPTION_STYLES = {
    'A':     dict(color='tab:blue',   ls='-',  label='A (H=+1 Dini, J_m)'),
    'C':     dict(color='tab:orange', ls='--', label=r'C (Dir/Robin, $J_{|m-1|}$)'),
    'G':     dict(color='tab:green',  ls=':',  label=r'G ($\psi$ Dirichlet, MMT)'),
    'G_old': dict(color='tab:olive',  ls='-.', label=r'G-old ($\psi$ Dirichlet, GL quad)'),
}

# Per-quantity colours used WITHIN a panel so that multiple traces from the
# same option remain visually distinguishable.  When multiple options are
# overlaid, linestyle (from OPTION_STYLES) still differentiates methods.
QUANTITY_COLORS = dict(
    KE_total='black',
    KE_low='tab:blue',
    KE_band='tab:red',
    KE_high='tab:green',
    cum_D_nu='tab:purple',
    cum_D_alpha='tab:cyan',
    net_loss='tab:gray',
    data='black',
    fit_1exp='tab:orange',
    fit_2exp='tab:red',
    residual='tab:purple',
    steady_mean='tab:olive',
)

# ---------------------------------------------------------------------------
# Filename / metadata parsing
# ---------------------------------------------------------------------------

def _parse_dedalus_num(s):
    """Parse strings like '1d4ep01' = 14, '4em04' = 4e-4."""
    first = float(s[0])
    try:
        sec = float(s[2])
    except Exception:
        sec = 0
    sgn = 1 if s[-3] == 'p' else -1
    exp = int(s[-2:])
    return (first + sec / 10) * 10 ** (sgn * exp)


def parse_params_from_filename(filename):
    """Pull (k_force, alpha, nu, eps, ...) from a processed-spectra .npy stem."""
    stem = Path(filename).stem
    out = {}
    for key in ['nu', 'gam', 'kf', 'ki', 'eps', 'alpha']:
        try:
            tok = stem.split(f'{key}_')[1].split('_')[0]
            out[key] = float(_parse_dedalus_num(tok))
        except Exception:
            out[key] = None
    # Useful derived quantities: forcing-band edges in physical wavenumber units
    # (matching the IVP's make_transform definition where  k_physical = k_force * pi).
    if out['kf'] is not None:
        out['k_band_low'] = out['kf'] * np.pi
        out['k_band_high'] = (out['kf'] + 1) * np.pi

        #out['kf'] /= 2
        #out['k_band_low'] /= 2
        #out['k_band_high'] /= 2    
        #print(out['kf'], out['k_band_low'], out['k_band_high'])

    return out

# ---------------------------------------------------------------------------
# Option detection
# ---------------------------------------------------------------------------

def detect_option(data):
    """Identify which post-processing produced this .npy."""
    keys = set(data.keys())
    if ('E_psiBn' in keys) or ('E_psiBn_tavg' in keys):
        return 'G_old'   # original grid-space Gauss-Legendre version of G
    if ('lambda_R' in keys) or ('keBn_r' in keys) or ('keBn_r_tavg' in keys):
        return 'C'
    if 'std_zs' in keys:
        return 'G'       # new analytical-Bouwkamp option G
    if ('dini_zs' in keys) or ('H' in keys):
        return 'A'
    raise ValueError("Cannot auto-detect option from .npy keys")

# ---------------------------------------------------------------------------
# Canonical-name extraction (matches plot_spectral_budget conventions)
# ---------------------------------------------------------------------------

def get_arrays(data, option):
    """Return canonical (E_bn, T_bn, D_nu_bn, D_alpha_bn, dE_dt_bn, tw, bin_centers,
    bin_widths) regardless of which option produced the file.  All *_bn arrays
    are shape (Nt, Nbins).
    """
    bin_centers = data['bin_centers']
    bin_edges   = data['bin_edges']
    bin_widths  = np.diff(bin_edges)
    tw          = np.atleast_1d(data['ts'])

    if option == 'G_old':
        E_bn       = data['E_psiBn']
        T_bn       = data['T_psiBn']
        D_nu_bn    = data['D_nu_Bn']
        D_alpha_bn = data['D_alpha_Bn']
        dE_dt_bn   = data.get('dE_psiBn_dt', np.zeros_like(E_bn))
    else:  # A, C, G all use the same naming after our convergence on conventions
        E_bn       = data['keBn']
        T_bn       = data['tBn']
        D_nu_bn    = data['D_nu_Bn']
        D_alpha_bn = data['D_alpha_Bn']
        dE_dt_bn   = data.get('dkeBn_dt', np.zeros_like(E_bn))

    return dict(E_bn=E_bn, T_bn=T_bn, D_nu_bn=D_nu_bn, D_alpha_bn=D_alpha_bn,
                dE_dt_bn=dE_dt_bn, tw=tw,
                bin_centers=bin_centers, bin_widths=bin_widths)

# ---------------------------------------------------------------------------
# Banding and integration helpers
# ---------------------------------------------------------------------------

def band_masks(bin_centers, k_band_low, k_band_high):
    """Return (mask_low, mask_band, mask_high) -- disjoint partitioning of
    the wavenumber bins by their centres against the IVP forcing band."""
    m_low  = bin_centers < k_band_low
    m_band = (bin_centers >= k_band_low) & (bin_centers < k_band_high)
    m_high = bin_centers >= k_band_high
    return m_low, m_band, m_high


def integrate_band(qbn, bin_widths, mask):
    """integral over the selected bins of qbn[:, b] * widths[b].  Shape: (Nt,)."""
    return np.sum(qbn[:, mask] * bin_widths[mask], axis=1)


def integrate_full(qbn, bin_widths):
    return np.sum(qbn * bin_widths, axis=1)


def cumulative_trapezoid(y, t):
    """Cumulative trapezoidal integration of y(t).  Returns array same shape as y,
    starting at 0."""
    out = np.zeros_like(y, dtype=float)
    if len(t) < 2:
        return out
    dt = np.diff(t)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt)
    return out

# ---------------------------------------------------------------------------
# Diagnostics computation
# ---------------------------------------------------------------------------

def compute_diagnostics(data, option, params, t_steady_frac=0.5):
    """Compute all derived diagnostic quantities for a single processed file."""
    arrs = get_arrays(data, option)
    tw          = arrs['tw']
    bin_centers = arrs['bin_centers']
    bin_widths  = arrs['bin_widths']
    E_bn        = arrs['E_bn']
    T_bn        = arrs['T_bn']
    D_nu_bn     = arrs['D_nu_bn']
    D_alpha_bn  = arrs['D_alpha_bn']
    dE_dt_bn    = arrs['dE_dt_bn']

    Nt = len(tw)

    # Build band masks
    if params['k_band_low'] is not None:
        m_low, m_band, m_high = band_masks(
            bin_centers, params['k_band_low'], params['k_band_high']
        )
    else:
        # If we can't parse k_force, fall back to a single all-band partition
        m_low  = np.zeros_like(bin_centers, dtype=bool)
        m_band = np.ones_like(bin_centers, dtype=bool)
        m_high = np.zeros_like(bin_centers, dtype=bool)

    # Total and band-integrated KE traces
    KE_total = integrate_full(E_bn, bin_widths)
    KE_low   = integrate_band(E_bn, bin_widths, m_low)
    KE_band  = integrate_band(E_bn, bin_widths, m_band)
    KE_high  = integrate_band(E_bn, bin_widths, m_high)

    # Verify disjoint: (KE_low + KE_band + KE_high) should equal KE_total
    disjoint_check = np.max(np.abs(KE_low + KE_band + KE_high - KE_total))

    # Total-dissipation traces (just spatial integrals; cumulate later)
    D_nu_total    = integrate_full(D_nu_bn,    bin_widths)
    D_alpha_total = integrate_full(D_alpha_bn, bin_widths)
    T_total       = integrate_full(T_bn,       bin_widths)
    dE_dt_total   = integrate_full(dE_dt_bn,   bin_widths)

    # Cumulative dissipation channels (trapezoidal in time)
    cum_D_nu    = cumulative_trapezoid(D_nu_total,    tw)
    cum_D_alpha = cumulative_trapezoid(D_alpha_total, tw)

    # Steady-state averaging window: last `t_steady_frac` of the time series
    n_steady = max(2, int(t_steady_frac * Nt))
    sl_steady = slice(Nt - n_steady, Nt)

    KE_initial = float(KE_total[0])
    KE_steady  = float(np.mean(KE_total[sl_steady]))
    R = KE_steady / KE_initial if abs(KE_initial) > 1e-30 else float('nan')

    # Fractional dissipation channels
    f_nu    = float(cum_D_nu[-1]    / KE_initial) if abs(KE_initial) > 1e-30 else float('nan')
    f_alpha = float(cum_D_alpha[-1] / KE_initial) if abs(KE_initial) > 1e-30 else float('nan')

    # Per-band budget integrals at steady state
    def bavg(qbn, mask):
        return float(np.mean(integrate_band(qbn, bin_widths, mask)[sl_steady]))

    T_low_ss      = bavg(T_bn,       m_low)
    T_band_ss     = bavg(T_bn,       m_band)
    T_high_ss     = bavg(T_bn,       m_high)
    Dnu_low_ss    = bavg(D_nu_bn,    m_low)
    Dnu_band_ss   = bavg(D_nu_bn,    m_band)
    Dnu_high_ss   = bavg(D_nu_bn,    m_high)
    Dal_low_ss    = bavg(D_alpha_bn, m_low)
    Dal_band_ss   = bavg(D_alpha_bn, m_band)
    Dal_high_ss   = bavg(D_alpha_bn, m_high)
    dEdt_low_ss   = bavg(dE_dt_bn,   m_low)
    dEdt_band_ss  = bavg(dE_dt_bn,   m_band)
    dEdt_high_ss  = bavg(dE_dt_bn,   m_high)

    # Residual = dE/dt - T + D_nu + D_alpha  (this should equal +F per band,
    # where F = eps in the forcing band and 0 elsewhere).  Sign convention:
    # ∂_t E = T - D_nu - D_alpha + F  =>  F = ∂_t E - T + D_nu + D_alpha = res.
    res_low_ss  = dEdt_low_ss  - T_low_ss  + Dnu_low_ss  + Dal_low_ss
    res_band_ss = dEdt_band_ss - T_band_ss + Dnu_band_ss + Dal_band_ss
    res_high_ss = dEdt_high_ss - T_high_ss + Dnu_high_ss + Dal_high_ss
    # Total (should also equal eps to within numerical residual)
    res_total_ss = (float(np.mean(dE_dt_total[sl_steady]))
                    - float(np.mean(T_total[sl_steady]))
                    + float(np.mean(D_nu_total[sl_steady]))
                    + float(np.mean(D_alpha_total[sl_steady])))

    # Exponential decay fits to KE_total(t) - KE_steady
    decay_y = KE_total - KE_steady
    # Fit only where decay_y > 0 (above the asymptote)
    valid = decay_y > 0
    # Need to avoid fitting too early in time (initial spike during
    # turbulence onset can look anomalous); skip the first snapshot.
    if Nt > 4:
        valid[:1] = False

    if np.sum(valid) >= 4:
        t_fit = tw[valid]
        y_fit = decay_y[valid]
        # Normalise t to start at 0 for numerical conditioning
        t0 = t_fit[0]
        try:
            popt1, _ = curve_fit(
                lambda t, A, r: A * np.exp(-r * (t - t0)),
                t_fit, y_fit,
                p0=[y_fit[0], 1.0 / max(t_fit[-1] - t0, 1e-3)],
                maxfev=4000,
            )
            r_1exp = float(abs(popt1[1]))
            y_pred1 = popt1[0] * np.exp(-popt1[1] * (t_fit - t0))
            ss_res1 = float(np.sum((y_fit - y_pred1) ** 2))
            n_pts   = len(y_fit)
            AIC_1exp = float(n_pts * np.log(ss_res1 / n_pts) + 2 * 2)
        except Exception:
            r_1exp, AIC_1exp = float('nan'), float('nan')

        try:
            popt2, _ = curve_fit(
                lambda t, A1, r1, A2, r2:
                    A1 * np.exp(-r1 * (t - t0)) + A2 * np.exp(-r2 * (t - t0)),
                t_fit, y_fit,
                p0=[0.5 * y_fit[0], 10.0 / max(t_fit[-1] - t0, 1e-3),
                    0.5 * y_fit[0], 1.0  / max(t_fit[-1] - t0, 1e-3)],
                maxfev=10000,
            )
            rates_2exp = sorted([abs(popt2[1]), abs(popt2[3])], reverse=True)
            r_2exp_fast, r_2exp_slow = float(rates_2exp[0]), float(rates_2exp[1])
            y_pred2 = (popt2[0] * np.exp(-popt2[1] * (t_fit - t0))
                       + popt2[2] * np.exp(-popt2[3] * (t_fit - t0)))
            ss_res2 = float(np.sum((y_fit - y_pred2) ** 2))
            n_pts   = len(y_fit)
            AIC_2exp = float(n_pts * np.log(ss_res2 / n_pts) + 2 * 4)
        except Exception:
            r_2exp_fast, r_2exp_slow, AIC_2exp = float('nan'), float('nan'), float('nan')
    else:
        r_1exp = r_2exp_fast = r_2exp_slow = AIC_1exp = AIC_2exp = float('nan')
        popt1 = popt2 = None
        t_fit = y_fit = None
        t0 = None

    return dict(
        tw=tw, KE_total=KE_total, KE_low=KE_low, KE_band=KE_band, KE_high=KE_high,
        D_nu_total=D_nu_total, D_alpha_total=D_alpha_total,
        T_total=T_total, dE_dt_total=dE_dt_total,
        cum_D_nu=cum_D_nu, cum_D_alpha=cum_D_alpha,
        disjoint_check=float(disjoint_check),
        KE_initial=KE_initial, KE_steady=KE_steady, R=R,
        f_nu=f_nu, f_alpha=f_alpha,
        res_low_ss=res_low_ss, res_band_ss=res_band_ss, res_high_ss=res_high_ss,
        res_total_ss=res_total_ss,
        T_low_ss=T_low_ss, T_band_ss=T_band_ss, T_high_ss=T_high_ss,
        Dnu_low_ss=Dnu_low_ss, Dnu_band_ss=Dnu_band_ss, Dnu_high_ss=Dnu_high_ss,
        Dal_low_ss=Dal_low_ss, Dal_band_ss=Dal_band_ss, Dal_high_ss=Dal_high_ss,
        r_1exp=r_1exp, r_2exp_fast=r_2exp_fast, r_2exp_slow=r_2exp_slow,
        AIC_1exp=AIC_1exp, AIC_2exp=AIC_2exp,
        # cached for plot use
        _decay_y=decay_y, _valid=valid,
        _popt1=popt1, _popt2=popt2,
        _t_fit=t_fit, _y_fit=y_fit, _t0=t0,
        # cached for budget residual time-series plot
        _T_bn=T_bn, _Dnu_bn=D_nu_bn, _Dal_bn=D_alpha_bn, _dEdt_bn=dE_dt_bn,
        _mask_band=m_band, _bw=bin_widths,
    )

# ---------------------------------------------------------------------------
# Figure construction (2x2)
# ---------------------------------------------------------------------------

def _format_method_label(opt, n_methods):
    """When only one method is loaded, label series by quantity alone.
    When multiple methods, prefix with the method tag.
    """
    if n_methods <= 1:
        return ''
    return f'{opt}: '


def _robust_ylim_around(value_traces, focus_value=None, frac=0.5, pad=0.15):
    """Pick a y-axis range that focuses on the long-time / steady part of
    `value_traces` (a list of 1D arrays).  Optionally pad symmetrically
    around `focus_value`.  Returns (ymin, ymax)."""
    arrs = []
    for y in value_traces:
        if y is None or len(y) == 0:
            continue
        n = len(y)
        i0 = int(frac * n)
        arrs.append(np.asarray(y[i0:]))
    if not arrs:
        return None
    cat = np.concatenate(arrs)
    finite = cat[np.isfinite(cat)]
    if len(finite) == 0:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if focus_value is not None:
        # Make sure focus_value is in view and the range encompasses both
        # sides symmetrically.
        lo = min(lo, focus_value)
        hi = max(hi, focus_value)
    span = hi - lo
    if span < 1e-12:
        span = max(1.0, abs(focus_value or 1.0))
    pad_amt = pad * span
    return lo - pad_amt, hi + pad_amt


def make_diagnostics_figure(datasets, options, params, out_stem,
                            t_steady_frac=0.5, eps_expected=1.0):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    ax_KE       = axes[0, 0]
    ax_cumdiss  = axes[0, 1]
    ax_decay    = axes[1, 0]
    ax_residual = axes[1, 1]

    diags = []
    n_methods = len(options)

    # Collect series for axis-limit decisions
    residual_traces = []

    for data, opt in zip(datasets, options):
        d = compute_diagnostics(data, opt, params, t_steady_frac=t_steady_frac)
        diags.append((opt, d))
        style = OPTION_STYLES[opt]
        ls    = style['ls']
        mlbl  = _format_method_label(opt, n_methods)

        # --- KE traces ---
        # KE_total is the headline curve, drawn bold in black (or method's
        # base colour if multiple methods are being compared).
        col_total = style['color'] if n_methods > 1 else QUANTITY_COLORS['KE_total']
        ax_KE.plot(d['tw'], d['KE_total'],
                   color=col_total, ls=ls, lw=2.2,
                   label=f'{mlbl}KE total')
        ax_KE.plot(d['tw'], d['KE_low'],
                   color=QUANTITY_COLORS['KE_low'], ls=ls, lw=1.2, alpha=0.85,
                   label=f'{mlbl}KE low (k<k_f)')
        ax_KE.plot(d['tw'], d['KE_band'],
                   color=QUANTITY_COLORS['KE_band'], ls=ls, lw=1.2, alpha=0.85,
                   label=f'{mlbl}KE band (k_f<=k<k_f+pi)')
        ax_KE.plot(d['tw'], d['KE_high'],
                   color=QUANTITY_COLORS['KE_high'], ls=ls, lw=1.2, alpha=0.85,
                   label=f'{mlbl}KE high (k>=k_f+pi)')

        # --- Cumulative dissipation ---
        col_total = style['color'] if n_methods > 1 else QUANTITY_COLORS['net_loss']
        ax_cumdiss.plot(d['tw'], d['cum_D_nu'],
                        color=QUANTITY_COLORS['cum_D_nu'], ls=ls, lw=2.0,
                        label=f'{mlbl}' r'cum $D_\nu$')
        ax_cumdiss.plot(d['tw'], d['cum_D_alpha'],
                        color=QUANTITY_COLORS['cum_D_alpha'], ls=ls, lw=2.0,
                        label=f'{mlbl}' r'cum $D_\alpha$')
        ax_cumdiss.plot(d['tw'], d['KE_initial'] - d['KE_total'],
                        color=col_total, ls=ls, lw=1.4, alpha=0.85,
                        label=f'{mlbl}' r'$KE(0) - KE(t)$')

        # --- Decay diagnostic ---
        if d['_t_fit'] is not None and len(d['_t_fit']) > 3:
            ax_decay.plot(d['_t_fit'], d['_y_fit'],
                          color=QUANTITY_COLORS['data'], ls=ls, lw=2.0,
                          marker='.', markersize=3,
                          label=f'{mlbl}data')
            if d['_popt1'] is not None:
                y1 = d['_popt1'][0] * np.exp(-d['_popt1'][1] * (d['_t_fit'] - d['_t0']))
                ax_decay.plot(d['_t_fit'], y1,
                              color=QUANTITY_COLORS['fit_1exp'], ls=ls,
                              lw=1.4, alpha=0.85,
                              label=f"{mlbl}1-exp ($r=${d['r_1exp']:.3g})")
            if d['_popt2'] is not None:
                y2 = (d['_popt2'][0] * np.exp(-d['_popt2'][1] * (d['_t_fit'] - d['_t0']))
                      + d['_popt2'][2] * np.exp(-d['_popt2'][3] * (d['_t_fit'] - d['_t0'])))
                ax_decay.plot(d['_t_fit'], y2,
                              color=QUANTITY_COLORS['fit_2exp'], ls=ls,
                              lw=1.4, alpha=0.85,
                              label=f"{mlbl}2-exp (fast={d['r_2exp_fast']:.3g}, slow={d['r_2exp_slow']:.3g})")

        # --- Per-band budget residual (forcing band only) ---
        res_band_t = integrate_band(
            d['_dEdt_bn'] - d['_T_bn'] + d['_Dnu_bn'] + d['_Dal_bn'],
            d['_bw'], d['_mask_band'],
        )
        col_res = style['color'] if n_methods > 1 else QUANTITY_COLORS['residual']
        ax_residual.plot(d['tw'], res_band_t,
                         color=col_res, ls=ls, lw=1.6,
                         label=f'{mlbl}' r'res$_{band}(t)$')

        # Steady mean line per method
        ax_residual.axhline(d['res_band_ss'],
                            color=col_res, ls=':', lw=1.0, alpha=0.6,
                            label=f'{mlbl}steady ' r'$\varepsilon_{\rm inf}$' f"={d['res_band_ss']:.3g}")

        residual_traces.append(res_band_t)

    # --- Reference lines on the residual panel ---
    ax_residual.axhline(eps_expected, color='red',  lw=1.0, alpha=0.85,
                        label=rf'$\varepsilon_{{\rm expected}} = {eps_expected:g}$')
    ax_residual.axhline(0, color='black', lw=0.6, alpha=0.5)

    # --- Zoom the residual panel onto the steady-state region ---
    # Pick limits from the last `t_steady_frac` of the data, focused around
    # eps_expected.
    ylim = _robust_ylim_around(residual_traces, focus_value=eps_expected,
                                frac=1.0 - t_steady_frac, pad=0.30)
    if ylim is not None:
        ax_residual.set_ylim(*ylim)

    # Panel decorations
    ax_KE.set_xlabel(r'$t$')
    ax_KE.set_ylabel(r'$KE$')
    ax_KE.set_yscale('log')
    ax_KE.set_title('KE total (black) and disjoint band traces')
    ax_KE.legend(loc='best', fontsize=8)

    ax_cumdiss.set_xlabel(r'$t$')
    ax_cumdiss.set_ylabel('cumulative energy')
    ax_cumdiss.set_title(r'Cumulative dissipation channels and $KE(0)-KE(t)$')
    ax_cumdiss.legend(loc='best', fontsize=8)

    ax_decay.set_xlabel(r'$t$')
    ax_decay.set_ylabel(r'$KE_{tot}(t) - KE_{tot,\,steady}$')
    ax_decay.set_yscale('log')
    ax_decay.set_title('Decay fit')
    ax_decay.legend(loc='best', fontsize=8)

    ax_residual.set_xlabel(r'$t$')
    ax_residual.set_ylabel(r'$\partial_t E_{band} - T_{band} + D_\nu^{band} + D_\alpha^{band}$')
    ax_residual.set_title(r'Forcing-band budget residual ($\approx \varepsilon$)')
    ax_residual.legend(loc='best', fontsize=8)

    # Inline summary annotation (single-method case only; multi-method gets
    # noisy if we annotate per method).
    if n_methods == 1:
        d = diags[0][1]
        opt = diags[0][0]
        # Band-capture fraction: how much of the (k-integrated) inferred eps
        # lives in the band we've defined.  < 1 typically because the Cartesian
        # forcing-shell spread doesn't perfectly localise to one disk-basis
        # band (see physical explanation in code comments above).
        if abs(d['res_total_ss']) > 1e-30:
            band_capture_frac = d['res_band_ss'] / d['res_total_ss']
        else:
            band_capture_frac = float('nan')
        summary_lines = [
            f"R = {d['R']:.3g}",
            rf"$\varepsilon_{{\rm inf}}$ (total) = {d['res_total_ss']:.3g}   "
            rf"($\varepsilon$ = {eps_expected:g})",
            rf"$\varepsilon_{{\rm inf}}$ (band)  = {d['res_band_ss']:.3g}   "
            rf"(band cap = {band_capture_frac:.2f})",
            f"f_nu, f_alpha = {d['f_nu']:.3g}, {d['f_alpha']:.3g}",
            f"1-exp rate = {d['r_1exp']:.3g}",
            f"2-exp rates = {d['r_2exp_fast']:.3g}, {d['r_2exp_slow']:.3g}",
            rf"$\Delta$AIC (2-1) = {d['AIC_2exp'] - d['AIC_1exp']:.3g}",
        ]
        fig.text(0.985, 0.5, '\n'.join(summary_lines),
                 ha='right', va='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4',
                           facecolor='whitesmoke', edgecolor='gray', alpha=0.85),
                 transform=fig.transFigure)

    # Suptitle with parameters
    title_bits = []
    for k in ['alpha', 'nu', 'kf', 'eps']:
        v = params.get(k)
        if v is not None:
            title_bits.append(f"{k}={v:.4g}")
    fig.suptitle('Run diagnostics — ' + '   '.join(title_bits)
                 + f"\n[options: {''.join([o[0] for o in options])}]",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.96 if n_methods == 1 else 1.0, 0.96])

    return fig, diags

# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def write_json(diags, params, options, out_path):
    out = {
        'parameters': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in params.items()},
        'methods': {},
    }
    for (opt, d), _ in zip(zip(options, [x[1] for x in diags]), range(len(options))):
        pass  # placeholder if you want to add per-method blocks below
    for (opt, d) in diags:
        out['methods'][opt] = dict(
            KE_initial=d['KE_initial'],
            KE_steady=d['KE_steady'],
            R=d['R'],
            f_nu=d['f_nu'],
            f_alpha=d['f_alpha'],
            disjoint_check=d['disjoint_check'],
            res_low_ss=d['res_low_ss'],
            res_band_ss=d['res_band_ss'],
            res_high_ss=d['res_high_ss'],
            res_total_ss=d['res_total_ss'],
            T_low_ss=d['T_low_ss'], T_band_ss=d['T_band_ss'], T_high_ss=d['T_high_ss'],
            Dnu_low_ss=d['Dnu_low_ss'], Dnu_band_ss=d['Dnu_band_ss'], Dnu_high_ss=d['Dnu_high_ss'],
            Dal_low_ss=d['Dal_low_ss'], Dal_band_ss=d['Dal_band_ss'], Dal_high_ss=d['Dal_high_ss'],
            r_1exp=d['r_1exp'],
            r_2exp_fast=d['r_2exp_fast'],
            r_2exp_slow=d['r_2exp_slow'],
            AIC_1exp=d['AIC_1exp'],
            AIC_2exp=d['AIC_2exp'],
        )
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=lambda x: (
            float(x) if isinstance(x, (np.floating, np.integer)) else str(x)
        ))

# ---------------------------------------------------------------------------
# Output filename hygiene (mirrors plot_spectral_budget conventions)
# ---------------------------------------------------------------------------

def parse_output_suffix(filename):
    stem = Path(filename).stem
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files   = args['<file>']
    outdir  = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    t_steady_frac = float(args['--t_steady_frac'])
    no_json    = bool(args.get('--no_json'))
    no_figure  = bool(args.get('--no_figure'))

    datasets, options, suffixes = [], [], []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"ERROR: file not found: {f}", file=sys.stderr)
            sys.exit(1)
        data = np.load(path, allow_pickle=True).item()
        opt  = detect_option(data)
        datasets.append(data)
        options.append(opt)
        suffixes.append(parse_output_suffix(f))
        print(f"Loaded {path.name}  ->  option {opt}")

    # Parse parameters from the first input file (all inputs should share the
    # same simulation parameters; the filenames typically match).
    params = parse_params_from_filename(files[0])
    eps_expected = params.get('eps', 1.0) or 1.0

    # Build output filename stem
    unique_suffixes = list(dict.fromkeys(suffixes))
    out_stem = unique_suffixes[0] if len(unique_suffixes) == 1 else '_'.join(unique_suffixes)
    options_tag = ''.join([o[0] for o in options])

    # Figure
    fig, diags = make_diagnostics_figure(
        datasets, options, params, out_stem,
        t_steady_frac=t_steady_frac, eps_expected=eps_expected,
    )
    if not no_figure:
        out_fig = outdir / f"diagnostics_{options_tag}_{out_stem}.png"
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"Wrote {out_fig}")
    else:
        plt.close(fig)

    # JSON
    if not no_json:
        out_json = outdir / f"diagnostics_{options_tag}_{out_stem}.json"
        write_json(diags, params, options, out_json)
        print(f"Wrote {out_json}")

    # Print a brief summary to stdout for at-the-terminal feedback
    print()
    print('=' * 72)
    print('Summary')
    print('=' * 72)
    for opt, d in diags:
        print(f'\n  Option {opt}:')
        print(f'    KE_initial      = {d["KE_initial"]:.4g}')
        print(f'    KE_steady       = {d["KE_steady"]:.4g}')
        print(f'    R = KE_s/KE_0   = {d["R"]:.4g}')
        print(f'    f_nu, f_alpha   = {d["f_nu"]:.4g}, {d["f_alpha"]:.4g}')
        print(f'    eps_inferred    = {d["res_band_ss"]:.4g}   (expected {eps_expected})')
        print(f'    residuals (l/b/h, tot) = '
              f'{d["res_low_ss"]:.3g}, {d["res_band_ss"]:.3g}, {d["res_high_ss"]:.3g}, '
              f'{d["res_total_ss"]:.3g}')
        print(f'    decay rates   : 1-exp={d["r_1exp"]:.3g}; '
              f'2-exp fast/slow={d["r_2exp_fast"]:.3g}/{d["r_2exp_slow"]:.3g}')
        print(f'    AIC: 1-exp={d["AIC_1exp"]:.3g}, 2-exp={d["AIC_2exp"]:.3g}')
        print(f'    disjoint_check  = {d["disjoint_check"]:.3g}   (should be ~0)')
    print()


if __name__ == '__main__':
    main()

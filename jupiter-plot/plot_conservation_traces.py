"""
plot_conservation_traces.py

Multi-panel time-series QA plot from an IVP analysis HDF5 file.

Reads scalar diagnostics saved by disk_turb_run_init_v3.py and plots:

  Row 1 (global health):
    KE(t)         total / area-averaged kinetic energy
    <omega>(t)    area-averaged vorticity
    enstrophy(t)  area-averaged omega^2

  Row 2 (linear & angular momentum conservation):
    L(t)          total angular momentum (integral of r * u_phi)
                  -- isolates the m=0 H=-1 Dini initial term exactly;
                     should remain at zero by construction
    P_x(t)        total linear momentum (x-component)
    P_y(t)        total linear momentum (y-component)
                  -- not strictly conserved (pressure on the wall) but should
                     fluctuate small around zero with the symmetric forcing/IC

  Row 3 (initial-term diagnostics for the H=-1 Dini expansion of u_phi):
    D_init_1_cos, D_init_1_sin       m=1 sector, basis r^0=1
    D_init_2_cos, D_init_2_sin       m=2 sector, basis r^1=r
    log| all initial-term modes |    combined view on log-y, including
                                     |L|, |Px|, |Py| scaled to a common
                                     reference where possible

Designed for quick visual QA on each parameter-study run.

The script gracefully handles older runs that don't have all of L/Px/Py
or D_init_*; missing panels show "n/a" rather than crashing.

Usage:
    plot_conservation_traces.py <hdf5_file> [options]

Options:
    --outdir=<path>     Output directory [default: .]
    --logy_combined     Use log-y for the combined initial-term panel [default: True]
"""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
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
# Loading helpers
# ---------------------------------------------------------------------------

def safe_load(f, taskname):
    """Return (t, values) for the scalar task `taskname` from a Dedalus
    analysis file `f`, or (None, None) if the task is absent.
    Scalar tasks are stored with shape (Nt, 1, 1, ...); we squeeze to (Nt,).
    """
    if 'tasks' not in f or taskname not in f['tasks']:
        return None, None
    dset = f['tasks'][taskname]
    arr  = np.squeeze(np.array(dset))
    # Try to pull the sim_time scale; fall back to integer indices.
    try:
        t = np.array(dset.dims[0]['sim_time'])
    except Exception:
        t = np.arange(arr.shape[0])
    return t, arr


def _plot_zero_centered(ax, t, y, label=None, color=None):
    """Helper for L, Px, Py, D_init_*: linear y, zero line, single trace."""
    if t is None:
        ax.text(0.5, 0.5, 'n/a (not saved)', transform=ax.transAxes,
                ha='center', va='center', color='gray', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.plot(t, y, color=color, lw=1.5, label=label)
    ax.axhline(0, color='black', lw=0.5, alpha=0.4)


def _safe_max_abs(*arrs):
    """Maximum of |arr| over all arrays, ignoring None entries."""
    vals = []
    for a in arrs:
        if a is not None:
            vals.append(np.max(np.abs(a)))
    return max(vals) if vals else 0.0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    file_str = args['<hdf5_file>']
    outdir   = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)

    path = Path(file_str)
    if not path.exists():
        print(f"ERROR: file not found: {file_str}", file=sys.stderr)
        sys.exit(1)

    f = h5py.File(file_str, 'r')

    # --- Load each scalar task, gracefully tolerating missing ones ---
    fields = {}
    for name in ['KE', 'W', 'EN',
                 'L', 'Px', 'Py',
                 'D_init_1_cos', 'D_init_1_sin',
                 'D_init_2_cos', 'D_init_2_sin']:
        fields[name] = safe_load(f, name)

    print(list(f['tasks']))

    KE_t, KE   = fields['KE']
    W_t,  W    = fields['W']
    EN_t, EN   = fields['EN']
    L_t,  L    = fields['L']
    Px_t, Px   = fields['Px']
    Py_t, Py   = fields['Py']
    D1c_t, D1c = fields['D_init_1_cos']
    D1s_t, D1s = fields['D_init_1_sin']
    D2c_t, D2c = fields['D_init_2_cos']
    D2s_t, D2s = fields['D_init_2_sin']

    # --- 3x3 grid ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    # ---------- Row 1: global health ----------
    if KE is not None:
        axes[0, 0].plot(KE_t, KE, color='tab:blue', lw=1.5)
        axes[0, 0].set_ylabel('KE (area-avg)')
        axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(r'$KE = \langle\frac{1}{2}|u|^2\rangle$')

    if W is not None:
        axes[0, 1].plot(W_t, W, color='tab:purple', lw=1.5)
        axes[0, 1].axhline(0, color='black', lw=0.5, alpha=0.4)
        axes[0, 1].set_ylabel(r'$\langle\omega\rangle$')
    axes[0, 1].set_title(r'$\langle\omega\rangle$')

    if EN is not None:
        axes[0, 2].plot(EN_t, EN, color='tab:red', lw=1.5)
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_ylabel(r'$\langle\omega^2\rangle$')
    axes[0, 2].set_title(r'Enstrophy $\langle\omega^2\rangle$')

    # ---------- Row 2: linear & angular momentum ----------
    _plot_zero_centered(axes[1, 0], L_t, L, color='tab:green')
    axes[1, 0].set_ylabel(r'$L = \int r\,u_\phi\,dA$')
    axes[1, 0].set_title(r'Angular momentum  $L(t)$')

    _plot_zero_centered(axes[1, 1], Px_t, Px, color='tab:orange')
    axes[1, 1].set_ylabel(r'$P_x = \int u_x\,dA$')
    axes[1, 1].set_title(r'Linear momentum  $P_x(t)$')

    _plot_zero_centered(axes[1, 2], Py_t, Py, color='tab:orange')
    axes[1, 2].set_ylabel(r'$P_y = \int u_y\,dA$')
    axes[1, 2].set_title(r'Linear momentum  $P_y(t)$')

    # ---------- Row 3: initial-term diagnostics ----------
    # Panel (2,0): m=1 sector, cos & sin overlaid
    if D1c is not None or D1s is not None:
        if D1c is not None:
            axes[2, 0].plot(D1c_t, D1c, color='tab:cyan', lw=1.5,
                            label=r'$D_{1,\cos}$')
        if D1s is not None:
            axes[2, 0].plot(D1s_t, D1s, color='tab:purple', lw=1.5,
                            label=r'$D_{1,\sin}$')
        axes[2, 0].axhline(0, color='black', lw=0.5, alpha=0.4)
        axes[2, 0].legend(loc='best')
    else:
        axes[2, 0].text(0.5, 0.5, 'n/a (not saved)',
                        transform=axes[2, 0].transAxes,
                        ha='center', va='center', color='gray', style='italic')
        axes[2, 0].set_xticks([]); axes[2, 0].set_yticks([])
    axes[2, 0].set_ylabel('m=1 init coef')
    axes[2, 0].set_title(r'$m=1$ initial-term ($\propto$ uniform translation in $u_\phi$)')

    # Panel (2,1): m=2 sector, cos & sin overlaid
    if D2c is not None or D2s is not None:
        if D2c is not None:
            axes[2, 1].plot(D2c_t, D2c, color='tab:cyan', lw=1.5,
                            label=r'$D_{2,\cos}$')
        if D2s is not None:
            axes[2, 1].plot(D2s_t, D2s, color='tab:purple', lw=1.5,
                            label=r'$D_{2,\sin}$')
        axes[2, 1].axhline(0, color='black', lw=0.5, alpha=0.4)
        axes[2, 1].legend(loc='best')
    else:
        axes[2, 1].text(0.5, 0.5, 'n/a (not saved)',
                        transform=axes[2, 1].transAxes,
                        ha='center', va='center', color='gray', style='italic')
        axes[2, 1].set_xticks([]); axes[2, 1].set_yticks([])
    axes[2, 1].set_ylabel('m=2 init coef')
    axes[2, 1].set_title(r'$m=2$ initial-term ($u_\phi$ linear-in-$r$ at $m=2$)')

    # Panel (2,2): combined log-|values| view of all "should-be-zero"
    # quantities, normalized to make them comparable.  The normalization:
    # divide each by sqrt(KE * disk_area), which roughly gives a velocity-scale
    # reference (disk_area = pi for unit disk).  If everything stays well below
    # 1 in these units, the initial-term content is dynamically negligible.
    ax = axes[2, 2]
    have_any = False
    if KE is not None and len(KE) > 0:
        # Build a smoothed sqrt(KE * pi) reference for normalization, using a
        # shared time axis (use KE's t).
        urms_ref = np.sqrt(np.maximum(KE, 1e-30) * np.pi)

        def _trace(name, t_, y_, color, ls='-'):
            nonlocal have_any
            if t_ is None or y_ is None:
                return
            # Interpolate the reference to t_ if different.
            if not np.array_equal(t_, KE_t):
                u_ref = np.interp(t_, KE_t, urms_ref)
            else:
                u_ref = urms_ref
            with np.errstate(divide='ignore', invalid='ignore'):
                yn = np.abs(y_) / np.maximum(u_ref, 1e-30)
            ax.plot(t_, np.maximum(yn, 1e-30),
                    color=color, ls=ls, lw=1.2, label=name)
            have_any = True

        _trace(r'$|L|/\sqrt{KE}$',         L_t,  L,   'tab:green')
        _trace(r'$|P_x|/\sqrt{KE}$',        Px_t, Px,  'tab:orange')
        _trace(r'$|P_y|/\sqrt{KE}$',        Py_t, Py,  'tab:orange', ls='--')
        _trace(r'$|D_{1,\cos}|/\sqrt{KE}$', D1c_t, D1c, 'tab:cyan')
        _trace(r'$|D_{1,\sin}|/\sqrt{KE}$', D1s_t, D1s, 'tab:cyan', ls='--')
        _trace(r'$|D_{2,\cos}|/\sqrt{KE}$', D2c_t, D2c, 'tab:purple')
        _trace(r'$|D_{2,\sin}|/\sqrt{KE}$', D2s_t, D2s, 'tab:purple', ls='--')

    if have_any:
        ax.set_yscale('log')
        ax.legend(loc='best', ncol=2)
        ax.axhline(1.0, color='red', lw=0.5, alpha=0.3)  # "things are large" reference
    else:
        ax.text(0.5, 0.5, 'n/a (no D_init or L/P data)',
                transform=ax.transAxes,
                ha='center', va='center', color='gray', style='italic')
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(r'$|x|/\sqrt{KE \cdot A_{\rm disk}}$')
    ax.set_title('Combined "should be ≪ 1" check')

    # x label only on bottom row
    for ax in axes[2, :]:
        ax.set_xlabel(r'$t$')

    # ----- title & save -----
    stem = path.stem.replace('analysis_', '')
    fig.suptitle(f'Conservation & initial-term traces — {stem}',
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = outdir / f'conservation_{stem}.pdf'
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

"""
plot_vorticity_mosaic.py

Snapshot mosaic of vorticity omega(r, phi, t) at user-specified times,
for quick visual confirmation that:

  * Early-time snapshots look numerically resolved (no obvious gridding
    artifacts from sub-Nyquist scales during the rapid turbulence onset).
  * Mid/late-time snapshots show the expected large-scale structures from
    the inverse cascade of energy from k_init to the smallest available
    wavenumbers (the condensate).

Defaults to a time list that emphasizes the early transient (where the
fast time-scale turbulence develops and the resolution check is most
informative) plus a couple of late frames to confirm the condensate is
present.

Loads the `vort` task from the IVP analysis HDF5 file (saved with
layout='g').  Each panel is a polar pcolormesh of omega(r, phi) at the
selected time.  Color range is set symmetrically about zero, using the
peak |omega| across the selected snapshots so all panels share a
consistent dynamic range.

Usage:
    plot_vorticity_mosaic.py <hdf5_file> [options]

Options:
    --times=<list>      Comma-separated list of times (or the literal
                        "end" for the final saved snapshot) [default: 0,0.05,0.1,0.25,0.5,1.0,5.0,end]
    --outdir=<path>     Output directory [default: .]
    --cmap=<str>        Matplotlib colormap [default: RdBu_r]
    --vmax=<float>      Override the auto symmetric color range with this
                        value (use the same vmax on every panel) [default: auto]
"""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path
import sys

from docopt import docopt
args = docopt(__doc__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

mpl.rcParams['figure.dpi']  = 100
mpl.rcParams['savefig.dpi'] = 200
mpl.rcParams['axes.titlesize'] = 10

# ---------------------------------------------------------------------------
# Grid extraction
# ---------------------------------------------------------------------------

def get_grid(f, taskname):
    """Pull (t, phi, r) arrays from a Dedalus analysis file for `taskname`.
    Tries dimension-scale access first; falls back to constructing from
    inferred Nphi/Nr if not present.
    """
    dset = f['tasks'][taskname]
    t    = np.array(dset.dims[0]['sim_time'])

    # phi
    phi = None
    try:
        # Dedalus tags dimensions with a scale named "phi_hash_..." (or just
        # by index): try numeric index 0 of the first dim scale set.
        dim_phi = dset.dims[1]
        # Try direct keys first
        for key in dim_phi.keys():
            phi = np.array(dim_phi[key])
            break
    except Exception:
        phi = None
    if phi is None:
        Nphi = dset.shape[1]
        phi  = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

    # r
    r = None
    try:
        dim_r = dset.dims[2]
        for key in dim_r.keys():
            r = np.array(dim_r[key])
            break
    except Exception:
        r = None
    if r is None:
        # Fall back to Gauss-Legendre nodes mapped to [0, 1]
        Nr = dset.shape[2]
        # rough approximation, evenly spaced (only used if HDF5 didn't
        # save the scale -- shouldn't matter visually)
        r = (np.arange(Nr) + 0.5) / Nr

    return t, phi, r

# ---------------------------------------------------------------------------
# Time selection
# ---------------------------------------------------------------------------

def parse_times(spec, t_array):
    """Convert a comma-separated string like "0,0.05,1.0,end" into a list of
    snapshot indices (nearest match for numeric entries; last index for
    "end")."""
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if tok.lower() == 'end':
            out.append(len(t_array) - 1)
        else:
            t_target = float(tok)
            idx = int(np.argmin(np.abs(t_array - t_target)))
            out.append(idx)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            uniq.append(idx)
    return uniq

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def grid_layout(n):
    """Return (nrows, ncols) for n panels: ncols <= 4."""
    if n <= 1:
        return 1, 1
    if n <= 4:
        return 2, 2 if n > 2 else (1, n)[0]  # fallback handled below
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    if n <= 16:
        return 4, 4
    # general fallback
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def _grid_layout(n):
    """Cleaner version: ncols = min(4, n); nrows = ceil(n / ncols)."""
    ncols = min(4, n) if n > 0 else 1
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    file_str = args['<hdf5_file>']
    outdir   = Path(args['--outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    times_spec = args['--times']
    cmap_name  = args['--cmap']
    vmax_arg   = args['--vmax']

    path = Path(file_str)
    if not path.exists():
        print(f"ERROR: file not found: {file_str}", file=sys.stderr)
        sys.exit(1)

    f = h5py.File(file_str, 'r')
    if 'tasks' not in f or 'vort' not in f['tasks']:
        print(f"ERROR: 'vort' task not found in {file_str}", file=sys.stderr)
        sys.exit(1)

    t_arr, phi, r = get_grid(f, 'vort')
    vort_dset = f['tasks']['vort']

    idxs = parse_times(times_spec, t_arr)
    print(f"Selected {len(idxs)} snapshots, t = "
          + ', '.join(f'{t_arr[i]:.3f}' for i in idxs))

    # Snapshot data
    snapshots = [np.array(vort_dset[i]) for i in idxs]

    # Compute symmetric colour range across all selected snapshots.
    if vmax_arg.lower() == 'auto':
        vmax = max(np.abs(s).max() for s in snapshots)
    else:
        vmax = float(vmax_arg)
    if vmax == 0.0:
        vmax = 1.0   # avoid degenerate range
    thresh = 1e-2 * vmax
    scl = 1 
    normsl = mpl.colors.SymLogNorm(linthresh=thresh, linscale=scl, vmin=-vmax, vmax=vmax, base=10)

    # ----- close the phi-wraparound for a clean polar pcolormesh -----
    phi_closed = np.concatenate([phi, [2*np.pi + phi[0]]])
    # 2D mesh for pcolormesh (cell-corner convention; sizes should match
    # face count, so a closed phi grid of length Nphi+1 paired with an
    # extended r grid handles the corners correctly).
    # For simplicity, use the more forgiving non-corner-aligned form and
    # rely on pcolormesh's shading='auto'.
    phi_mesh, r_mesh = np.meshgrid(phi_closed, r, indexing='ij')

    # ----- layout -----
    nrows, ncols = _grid_layout(len(idxs))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.3 * ncols + 0.5, 3.3 * nrows),
        subplot_kw={'projection': 'polar'},
        squeeze=False,
    )

    for k_ax, (idx, vort2d) in enumerate(zip(idxs, snapshots)):
        i_row = k_ax // ncols
        j_col = k_ax %  ncols
        ax = axes[i_row, j_col]

        # Close the phi axis for the data, too.
        vort_closed = np.concatenate([vort2d, vort2d[:1, :]], axis=0)

        pcm = ax.pcolormesh(
            phi_mesh, r_mesh, vort_closed,
            cmap=cmap_name,
            shading='auto', norm=normsl
        )
        ax.set_title(f't = {t_arr[idx]:.3f}', pad=8)
        ax.set_rticks([])      # cleaner; the radius is just visually 0..1
        ax.set_xticks([])      # angular ticks are noise in a small panel
        ax.grid(False)
        ax.set_rlim(0, 1)

    # Hide any unused subplots
    for k_ax in range(len(idxs), nrows * ncols):
        i_row = k_ax // ncols
        j_col = k_ax %  ncols
        axes[i_row, j_col].axis('off')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    thresh = 1e-2 * vmax
    fig.colorbar(pcm, cax=cbar_ax, label=r'$\omega$', ticks=ticker.SymmetricalLogLocator(base=10, linthresh=thresh))

    stem = path.stem.replace('analysis_', '')
    fig.suptitle(f'Vorticity snapshots — {stem}', y=0.99)
    #fig.tight_layout(rect=[0, 0, 0.92, 0.97])

    out = outdir / f'vorticity_mosaic_{stem}.png'#.pdf'
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()

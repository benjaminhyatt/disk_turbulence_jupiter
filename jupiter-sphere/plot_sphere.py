"""
Plot sphere outputs.

Two rendering modes are supported:
  oblique   - single oblique 2D orthographic projection of the sphere;
              --flip swaps which pole is visible.
  birds_eye - two side-by-side bird's-eye panels, N pole dead center in the
              left panel and S pole dead center in the right. A small black
              circle is drawn near each pole at colatitude --pole_ring_colat.

Uses pcolormesh (2D) instead of plot_surface (3D) for dramatically faster
rendering on large grids (2048x1024 and above).

Usage:
    plot_sphere_v3.py <files>... [options]

Options:
    --output=<dir>             output directory [default: ./frames]
    --mode=<str>               oblique | birds_eye [default: oblique]
    --task=<str>               HDF5 task name [default: vorticity]
    --flip=<bool>              (oblique only) False: N visible; True: S visible [default: False]
    --pole_ring_colat=<float>  (birds_eye only) colatitude (rad) of the small black ring drawn around each pole [default: 0.0873]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _orthographic_project(phi, theta, elev_deg, azim_deg):
    """
    Return 2D orthographic projection coordinates (px, py) and a boolean mask
    of the visible hemisphere for a (Nphi x Ntheta) grid.

    Parameters
    ----------
    phi   : 1-D array, shape (Nphi,)
    theta : 1-D array, shape (Ntheta,)   -- colatitude, 0=N pole
    elev_deg, azim_deg : viewing angles (matplotlib convention)

    Returns
    -------
    px, py : 2-D arrays, shape (Nphi, Ntheta)
    visible : 2-D bool array, True where the point faces the camera
    """
    PHI, THETA = np.meshgrid(phi, theta, indexing='ij')

    # Cartesian on unit sphere
    sx = np.sin(THETA) * np.cos(PHI)
    sy = np.sin(THETA) * np.sin(PHI)
    sz = np.cos(THETA)

    # Camera direction (unit vector toward viewer)
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    cx = np.cos(elev) * np.cos(azim)
    cy = np.cos(elev) * np.sin(azim)
    cz = np.sin(elev)

    # Orthographic basis: right = azim+90°, up = camera × right
    rx = -np.sin(azim)
    ry =  np.cos(azim)
    rz = 0.0
    # up = c × r
    ux = cy * rz - cz * ry
    uy = cz * rx - cx * rz
    uz = cx * ry - cy * rx

    # Project onto (right, up) plane
    px = sx * rx + sy * ry + sz * rz
    py = sx * ux + sy * uy + sz * uz

    # Visibility: dot product with camera direction > 0
    dot = sx * cx + sy * cy + sz * cz
    visible = dot >= 0.0

    return px, py, visible


def main_oblique(filename, start, count, output, opts):
    """Save oblique orthographic 2D plots of specified task for given range of writes."""
    flip_opt = opts[0]
    task = opts[1] if len(opts) > 1 else 'vorticity'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)

    elev_deg = -30.0 if flip_opt else 30.0
    azim_deg = -60.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi   = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()

        px, py, visible = _orthographic_project(phi, theta, elev_deg, azim_deg)

        # Equator and pole marker rings as index masks (approximate, like v2)
        eq_idx   = theta.size // 2
        pole_idx_near = 3
        pole_idx_far  = theta.size - 4

        mesh = None
        for index in range(start, start + count):
            data = dset[index, :, :].copy()
            # Black out equator and pole rings in data by clamping to zero
            # (they'll be overplotted as lines instead)
            clim = float(np.max(np.abs(data)))
            if clim == 0.0:
                clim = 1.0
            norm = matplotlib.colors.Normalize(-clim, clim)

            # Mask back hemisphere
            plot_data = np.where(visible, data, np.nan)

            if index == start:
                mesh = ax.pcolormesh(px, py, plot_data,
                                     norm=norm, cmap=cmap,
                                     shading='auto', rasterized=True)
                # Equator ring
                eq_phi = np.linspace(0, 2 * np.pi, 512)
                eq_x_3d = np.sin(theta[eq_idx]) * np.cos(eq_phi)
                eq_y_3d = np.sin(theta[eq_idx]) * np.sin(eq_phi)
                eq_z_3d = np.cos(theta[eq_idx]) * np.ones_like(eq_phi)
                elev_r = np.radians(elev_deg)
                azim_r = np.radians(azim_deg)
                cx = np.cos(elev_r) * np.cos(azim_r)
                cy = np.cos(elev_r) * np.sin(azim_r)
                cz = np.sin(elev_r)
                rx = -np.sin(azim_r); ry = np.cos(azim_r); rz = 0.0
                ux = cy * rz - cz * ry; uy = cz * rx - cx * rz; uz = cx * ry - cy * rx
                eq_px = eq_x_3d * rx + eq_y_3d * ry + eq_z_3d * rz
                eq_py = eq_x_3d * ux + eq_y_3d * uy + eq_z_3d * uz
                eq_vis = eq_x_3d * cx + eq_y_3d * cy + eq_z_3d * cz >= 0
                # Only plot visible arc segments
                eq_px_vis = np.where(eq_vis, eq_px, np.nan)
                ax.plot(eq_px_vis, eq_py, 'k-', lw=0.6, zorder=3)
            else:
                mesh.set_array(plot_data.ravel())
                mesh.set_norm(norm)

            title = title_func(file['scales/sim_time'][index])
            fig.suptitle(title, x=0.1, y=0.95, ha='left')
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)

    plt.close(fig)


def main_birds_eye(filename, start, count, output, opts):
    """Two-panel bird's-eye pcolormesh plots of the N pole (left) and S pole (right).

    Looking straight down (N panel: elev=+90) and straight up (S panel: elev=-90).
    The orthographic projection collapses to:
      px = sin(theta) * cos(phi)
      py = sin(theta) * sin(phi)
    but sin(theta) = sin(pi - theta), so both hemispheres map to the same disk.
    We mask each panel to its own hemisphere to avoid the degeneracy:
      N panel: theta < pi/2  (colatitude in northern hemisphere)
      S panel: theta > pi/2  (colatitude in southern hemisphere)
    The S panel also mirrors py (Y -> -Y) so it appears as viewed from below.
    """
    pole_ring_colat = opts[0]
    task = opts[1] if len(opts) > 1 else 'vorticity'
    cmap = plt.cm.RdBu_r
    dpi = 100
    figsize = (12, 6.5)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)

    fig, (ax_N, ax_S) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.1, 0.06, 0.8, 0.025])

    with h5py.File(filename, mode='r') as file:
        dset = file['tasks'][task]
        phi   = dset.dims[1][0][:].ravel()   # (Nphi,)
        theta = dset.dims[2][0][:].ravel()   # (Ntheta,) colatitude

        PHI, THETA = np.meshgrid(phi, theta, indexing='ij')
        X   =  np.sin(THETA) * np.cos(PHI)
        Y   =  np.sin(THETA) * np.sin(PHI)   # N panel (looking down)
        Y_S = -np.sin(THETA) * np.sin(PHI)   # S panel (mirrored, looking up)

        # Hemisphere masks: shape (Nphi, Ntheta)
        mask_N = THETA >= np.pi / 2   # hide southern hemisphere in N panel
        mask_S = THETA <= np.pi / 2   # hide northern hemisphere in S panel

        # Pole rings
        phi_ring = np.linspace(0, 2 * np.pi, 256)
        r_ring = np.sin(pole_ring_colat)
        ring_x = r_ring * np.cos(phi_ring)
        ring_y = r_ring * np.sin(phi_ring)

        mesh_N = mesh_S = cbar = sm = None

        for index in range(start, start + count):
            data = dset[index, :, :]
            clim = float(np.max(np.abs(data)))
            if clim == 0.0:
                clim = 1.0
            norm = matplotlib.colors.Normalize(-clim, clim)

            # Apply hemisphere masks — nan hides the degenerate hemisphere
            data_N = np.where(mask_N, np.nan, data)
            data_S = np.where(mask_S, np.nan, data)

            if index == start:
                mesh_N = ax_N.pcolormesh(X, Y,   data_N, norm=norm, cmap=cmap,
                                         shading='auto', rasterized=True)
                mesh_S = ax_S.pcolormesh(X, Y_S, data_S, norm=norm, cmap=cmap,
                                         shading='auto', rasterized=True)
                for ax_p, title in ((ax_N, 'N pole'), (ax_S, 'S pole')):
                    ax_p.set_aspect('equal')
                    ax_p.axis('off')
                    ax_p.set_title(title)
                    ax_p.plot(ring_x, ring_y, 'k-', lw=0.8, zorder=3)
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                                    label=task)
            else:
                mesh_N.set_array(data_N.ravel())
                mesh_S.set_array(data_S.ravel())
                mesh_N.set_norm(norm)
                mesh_S.set_norm(norm)
                sm.set_norm(norm)
                cbar.update_normal(sm)

            fig.suptitle(title_func(file['scales/sim_time'][index]), y=0.97)
            savename = savename_func(file['scales/write_number'][index])
            fig.savefig(str(output.joinpath(savename)), dpi=dpi)

    plt.close(fig)


main = main_oblique

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    mode = args['--mode'].lower()
    task = args['--task']
    if mode == 'oblique':
        flip = eval(args['--flip'])
        pass_options = [flip, task]
        fn = main_oblique
    elif mode in ('birds_eye', 'bird', 'birdseye', 'birds-eye'):
        pole_ring_colat = float(args['--pole_ring_colat'])
        pass_options = [pole_ring_colat, task]
        fn = main_birds_eye
    else:
        raise ValueError(f"Unknown --mode='{mode}'; expected 'oblique' or 'birds_eye'.")

    output_path = pathlib.Path(args['--output']).absolute()
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], fn, output=output_path, opts=pass_options)

"""
Helper functions used by process_tracking_locator_sphere.

Kept in this module to declutter the main script.  All functions here are
pure (no side effects, no dependence on script-level globals); the main
script imports what it needs from this file.

Contents:
  - great_circle_distance      : angular distance between two sphere points
  - find_threshold_crossing    : smallest rho where |profile| crosses |target|
  - cpc_frame_to_lab           : rotate CPC-frame small-circle coords to lab
                                 (theta, phi)
  - find_max_abs_in_cap        : grid-point extremum of |vort| inside a cap
  - refine_extremum_via_spline : sub-grid (theta, phi, vort) refinement via
                                 a local RectSphereBivariateSpline +
                                 L-BFGS-B optimization
"""

import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
from scipy.optimize import minimize


def great_circle_distance(theta1, phi1, theta2, phi2):
    """Great-circle distance between two points on the unit sphere (rad)."""
    cos_d = (np.cos(theta1) * np.cos(theta2)
           + np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    return np.arccos(np.clip(cos_d, -1.0, 1.0))


def find_threshold_crossing(rho_grid, profile, target):
    """Smallest rho where |profile| drops below |target|, with sub-grid
    linear interp between adjacent points. NaN if no crossing."""
    prof, targ = np.abs(profile), np.abs(target)
    crossings = np.where(prof < targ)[0]
    if len(crossings) == 0 or crossings[0] == 0:
        return np.nan
    k = crossings[0]
    return rho_grid[k - 1] + (targ - prof[k - 1]) * (rho_grid[k] - rho_grid[k - 1]) / (prof[k] - prof[k - 1])


def cpc_frame_to_lab(theta_cpc, phi_cpc, rho_grid, alpha_grid):
    """
    Map CPC-frame (rho_gc, alpha) coords to lab-frame (theta, phi) on the
    sphere.  rho_gc is great-circle distance from the CPC center; alpha
    is azimuthal angle measured counter-clockwise from local east (viewed
    from outside the sphere).
    """
    # 2D meshes of CPC-frame coords; shape (n_rho, n_alpha)
    rho_md, alpha_md = np.meshgrid(rho_grid, alpha_grid, indexing='ij')

    # Orthonormal CPC-centered basis (each row is a 3-component vector):
    #   c_hat = outward normal at CPC, e_hat = local east, n_hat = local north
    c_hat = np.array([np.sin(theta_cpc) * np.cos(phi_cpc),
                      np.sin(theta_cpc) * np.sin(phi_cpc),
                      np.cos(theta_cpc)])
    e_hat = np.array([-np.sin(phi_cpc), np.cos(phi_cpc), 0.0])
    n_hat = np.array([-np.cos(theta_cpc) * np.cos(phi_cpc),
                      -np.cos(theta_cpc) * np.sin(phi_cpc),
                       np.sin(theta_cpc)])

    # The trailing length-1 axis from [:, :, None] lines up against the
    # length-3 axis of each basis vector during broadcasting, producing
    # the (n_rho, n_alpha, 3) output below.
    cos_rho   = np.cos(rho_md)[:, :, None]
    sin_rho   = np.sin(rho_md)[:, :, None]
    cos_alpha = np.cos(alpha_md)[:, :, None]
    sin_alpha = np.sin(alpha_md)[:, :, None]
    p_hat = (cos_rho * c_hat
           + sin_rho * (cos_alpha * e_hat + sin_alpha * n_hat))
    theta_lab = np.arccos(np.clip(p_hat[:, :, 2], -1.0, 1.0))
    phi_lab   = np.arctan2(p_hat[:, :, 1], p_hat[:, :, 0]) % (2 * np.pi)
    return theta_lab, phi_lab


def find_max_abs_in_cap(vort_field_g, cap_idxs):
    """Locate the grid point with the largest |vort| inside the cap.
    Returns (phi_idx, theta_idx_global, vort_value, sign)."""
    sub = vort_field_g[:, cap_idxs]
    flat = int(np.argmax(np.abs(sub)))
    phi_idx, theta_idx_in_cap = np.unravel_index(flat, sub.shape)
    theta_idx_global = int(cap_idxs[theta_idx_in_cap])
    val = float(vort_field_g[phi_idx, theta_idx_global])
    return int(phi_idx), theta_idx_global, val, int(np.sign(val))


def refine_extremum_via_spline(theta_1d, phi_1d, vort_field,
                                phi_idx, theta_idx,
                                local_size_phi=2, local_size_theta=3,
                                opt_tol=1e-6):
    """
    Refine a grid-point vorticity extremum to sub-grid accuracy by fitting
    a local RectSphereBivariateSpline around the candidate and optimizing
    via L-BFGS-B.  Cheaper per call than spectral interpolation, which
    matters when running over many thousands of frames.

    To avoid the phi=0 / phi=2*pi wrap-around discontinuity, phi values
    are shifted so that the candidate sits at phi=0 in the spline's frame;
    the shift is undone for the output.

    Parameters
    ----------
    theta_1d : (Ntheta,) array
        Lab-frame colatitude grid in ascending order (N -> S), in [0, pi].
    phi_1d : (Nphi,) array
        Lab-frame longitude grid in ascending order, in [0, 2*pi).
    vort_field : (Nphi, Ntheta) array
        Vorticity on the lab-frame grid, same theta orientation as theta_1d.
    phi_idx, theta_idx : int
        Indices of the candidate (grid-point) extremum, e.g. from
        find_max_abs_in_cap.
    local_size_phi, local_size_theta : int
        Half-window sizes (in grid points) for the local spline sub-mesh.
        Defaults give (2*l+1)x(2*l+1) = 5x7 points around the candidate.
    opt_tol : float
        Tolerance passed to scipy.optimize.minimize (L-BFGS-B).

    Returns
    -------
    theta_refined, phi_refined : float
        Refined (theta, phi) location, with theta in [0, pi] and phi in
        [0, 2*pi).
    vort_refined : float
        Signed vorticity value at the refined location.
    """
    Nphi   = len(phi_1d)
    Ntheta = len(theta_1d)
    dphi   = 2 * np.pi / Nphi

    # Sign of the extremum we're tracking.  L-BFGS-B is a minimizer, so
    # we minimize -sign * spl to find the extremum of the right sign.
    grid_val = float(vort_field[phi_idx, theta_idx])
    sign = 1.0 if grid_val >= 0 else -1.0

    # Local theta range (clipped to grid bounds; near-pole windows may
    # be asymmetric, which is fine).
    th_lo = max(0, theta_idx - local_size_theta)
    th_hi = min(Ntheta - 1, theta_idx + local_size_theta)
    theta_local = theta_1d[th_lo:th_hi + 1]

    # Local phi indices with wrap-around (e.g., near phi=0 the window
    # straddles both ends of the array).  The corresponding phi values
    # are centered on the candidate (candidate at phi=0 in this frame),
    # which is a strictly monotonic sequence in [-pi, pi] and avoids any
    # wrap-around discontinuity in the spline's phi axis.
    raw_phi_idxs       = np.arange(phi_idx - local_size_phi,
                                    phi_idx + local_size_phi + 1)
    phi_idxs_local     = raw_phi_idxs % Nphi
    phi_local_centered = (raw_phi_idxs - phi_idx) * dphi

    # Extract the local data block.  np.ix_ gives a (n_phi_local,
    # n_theta_local) result, which we transpose to (theta, phi) order for
    # RectSphereBivariateSpline.
    data_block = vort_field[np.ix_(phi_idxs_local,
                                     np.arange(th_lo, th_hi + 1))]
    data_for_spline = data_block.T  # (n_theta_local, n_phi_local)

    # Build the spherical-aware spline.  pole_continuity=True is safe even
    # when the window doesn't touch a pole (the condition only applies
    # at u=0 / u=pi, which the local optimizer does not query unless they
    # are inside the window's bounds).
    spl = RectSphereBivariateSpline(
        theta_local, phi_local_centered, data_for_spline,
        pole_continuity=True,
    )

    # Objective and gradient for L-BFGS-B
    def neg_signed_val(x):
        return -sign * float(spl(x[0], x[1]).ravel()[0])

    def neg_signed_jac(x):
        dth = float(spl(x[0], x[1], dtheta=1).ravel()[0])
        dph = float(spl(x[0], x[1], dphi=1).ravel()[0])
        return np.array([-sign * dth, -sign * dph])

    theta_bounds = (float(theta_local[0]), float(theta_local[-1]))
    phi_bounds   = (float(phi_local_centered[0]),
                    float(phi_local_centered[-1]))

    x0 = [float(theta_1d[theta_idx]), 0.0]
    res = minimize(neg_signed_val, x0, jac=neg_signed_jac,
                    method='L-BFGS-B',
                    bounds=[theta_bounds, phi_bounds], tol=opt_tol)

    theta_refined        = float(res.x[0])
    phi_refined_centered = float(res.x[1])
    phi_refined          = (phi_refined_centered + float(phi_1d[phi_idx])) % (2 * np.pi)
    # res.fun = -sign * spl(x_opt)  =>  spl(x_opt) = -sign * res.fun
    vort_refined         = -sign * float(res.fun)

    return theta_refined, phi_refined, vort_refined

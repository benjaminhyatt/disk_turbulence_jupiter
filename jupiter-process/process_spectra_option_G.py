"""
Streamfunction (option G) post-processing for 2D disk turbulence.

For each saved snapshot:
  1. Load u (vector) and vort (scalar = omega).
  2. Solve  lap(psi) = vort  with gauge  psi(r=1) = 0  via LBVP.  This makes
     psi exactly Dirichlet at the boundary by construction.
  3. Obtain  u @ grad(vort)  (= u . grad(omega)) as a scalar field, called
     'nonlin_omega' here:
        * If the HDF5 file contains a saved 'nonlin_omega' task, use it.
        * Otherwise compute it post-hoc from u and vort via
          (u @ d3.grad(vort)).evaluate().
  4. phi-FFT both psi and nonlin_omega on Dedalus's radial grid (via
     np.fft.rfft, RealFourier-convention-aware).
  5. Project both into the scalar Dirichlet J_m Bessel basis (the natural
     basis for psi given the gauge) using Dedalus's exact Gauss-Legendre
     radial quadrature weights (dedalus_sphere.zernike.quadrature) and
     precomputed B_m basis matrices.
  6. Form the spectra
        E_psi(m, n) = 1/2 * lambda_{m,n}^2 * |psi_hat_{m,n}|^2 * (norm)
        T_psi(m, n) = -<psi_hat, (u@grad omega)_hat>_{m,n} * (norm)
     Bin into shared k-grid.  In this gauge,
        D_alpha = 2*alpha*E_psi     (exact)
        D_nu    = 2*nu*lambda^2*E_psi (exact in this basis, because
                  omega = lap(psi) = -lambda^2 psi per mode)
     so the entire budget closes within the scalar Dirichlet basis.

This pipeline serves as an independent cross-basis benchmark for the
Route C (process_spectra_zb_mbin_v3.py) pipeline.  The two are wrong in
different places:
  * Route C: exact BC for u_r (Dirichlet), approximate for u_phi (Robin
    H=-1 strict only at axisymmetric modes), and the nonlinear-term
    boundary trace is approximate (centrifugal -u_phi^2 at r=1).
  * Option G: exact BC for psi (Dirichlet by gauge), but loses the
    boundary trace of omega (since omega|_{r=1} = 2*u_phi|_{r=1} != 0 in
    general, whereas the psi-reconstructed omega = lap(psi) vanishes at
    the wall).
Agreement across the inertial range is strong evidence that both capture
the same physical spectral structure.

Usage:
    process_spectra_option_G.py <file>... [options]

Options:
    --t_out_start=<float>       Simulation time to begin making spectra [default: 0.]
    --t_out_end=<float>         Simulation time to stop making spectra [default: 100.]
    --t_steady_range=<float>    Size of time window prior to t_out_end to average as steady state [default: 50.]
    --steady_only=<bool>        True: save only time-averaged steady-state data; False: save all timesteps [default: True]
"""

import numpy as np
import h5py
import scipy.special as sp
import dedalus.public as d3
from dedalus.libraries.dedalus_sphere import zernike as zsphere
from mpi4py import MPI
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from docopt import docopt
args = docopt(__doc__)
if rank == 0:
    print(args)

# ---------------------------------------------------------------------------
# Parameter parsing from output filename (same convention as v2/v3)
# ---------------------------------------------------------------------------

def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2])
    except Exception:
        sec = 0
    sgn = 1 if a[-3] == 'p' else -1
    exp = int(a[-2:])
    return (first + sec / 10) * 10 ** (sgn * exp)

file_str = args['<file>'][0]
output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0]
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr   = int(output_suffix.split('Nr_')[1].split('_')[0])

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
eps_str   = output_suffix.split('eps_')[1].split('_')[0]
nu_str    = output_suffix.split('nu_')[1].split('_')[0]
kf_str    = output_suffix.split('kf_')[1].split('_')[0]

alpha_vals = np.array([1e-2, 2e-2, 4e-2])
gamma_vals = np.array([0])
eps_vals   = np.array([1.0])
nu_vals    = np.array([1e-4, 2e-4, 4e-4])
kf_vals    = np.array([10, 15, 20])

alpha   = alpha_vals[np.argmin(np.abs(alpha_vals - str_to_float(alpha_str)))]
gamma   = gamma_vals[np.argmin(np.abs(gamma_vals - str_to_float(gamma_str)))]
eps     = eps_vals[np.argmin(np.abs(eps_vals     - str_to_float(eps_str)))]
nu      = nu_vals[np.argmin(np.abs(nu_vals       - str_to_float(nu_str)))]
k_force = kf_vals[np.argmin(np.abs(kf_vals       - str_to_float(kf_str)))]

t_out_start    = float(args['--t_out_start'])
t_out_end      = float(args['--t_out_end'])
t_steady_range = float(args['--t_steady_range'])
steady_only    = eval(args['--steady_only'])

output_prefix = 'processed_spectra_optionG'
output_prefix += '_steady' if steady_only else ''

# ---------------------------------------------------------------------------
# Scalar Dirichlet helpers
# ---------------------------------------------------------------------------

def dirichlet_roots(m, N):
    """First N positive roots of J_m(z) = 0."""
    return sp.jn_zeros(m, N)

def dirichlet_weight(m, z):
    """Fourier-Bessel (Dirichlet) normalisation:
       int_0^1 J_m(z r)^2 r dr = (1/2) J_{m+1}(z)^2   for z a root of J_m.
    """
    return 0.5 * sp.jv(m + 1, z) ** 2

# ---------------------------------------------------------------------------
# phi-FFT helper.  Dedalus RealFourier convention:
#     f(phi) = a_0 + sum_{m>=1} [ a_m cos(m phi) - c_m sin(m phi) ]
# We work with standard amplitudes
#     f(phi) = a_0 + sum_{m>=1} [ a_m cos(m phi) + b_m sin(m phi) ],
# i.e. b_m = -c_m.  Conversion from np.fft.rfft output F[m]:
#     a_0  = (1/Nphi) F[0]
#     a_m  = (2/Nphi) Re(F[m])     (m=1..Nphi/2-1)
#     b_m  = -(2/Nphi) Im(F[m])    (m=1..Nphi/2-1)
# ---------------------------------------------------------------------------

def phi_fft_to_cos_sin(grid_array, Nphi):
    """grid_array shape (Nphi, Nr) -> (a, b) each of shape (Nm, Nr), where
    Nm = Nphi/2 + 1.  a, b are the cos and sin amplitudes."""
    F = np.fft.rfft(grid_array, axis=0)
    a = np.zeros_like(F.real)
    b = np.zeros_like(F.real)
    a[0] = F[0].real / Nphi
    Nm = F.shape[0]
    if Nphi % 2 == 0 and Nm == Nphi // 2 + 1:
        a[1:-1] = 2.0 * F[1:-1].real / Nphi
        b[1:-1] = -2.0 * F[1:-1].imag / Nphi
        a[-1]   = F[-1].real / Nphi
    else:
        a[1:] = 2.0 * F[1:].real / Nphi
        b[1:] = -2.0 * F[1:].imag / Nphi
    return a, b

# ---------------------------------------------------------------------------
# Setup: Bessel root tables for scalar Dirichlet at each azimuthal m
# ---------------------------------------------------------------------------

mmax = int(Nphi / 2) - 1
ms = np.arange(mmax + 1)

if rank == 0:
    logger.info("Computing Dirichlet root tables for m in 0..%d", mmax)

dirichlet_zs = {}
dirichlet_ws = {}
for m in ms:
    dirichlet_zs[m] = dirichlet_roots(m, Nr)
    dirichlet_ws[m] = dirichlet_weight(m, dirichlet_zs[m])

# Per-(m, n) eigenvalue / weight arrays.
lambda_arr = np.zeros((mmax + 1, Nr))
weight_arr = np.zeros((mmax + 1, Nr))
for m in ms:
    lambda_arr[m, :] = dirichlet_zs[m]
    weight_arr[m, :] = dirichlet_ws[m]

# ---------------------------------------------------------------------------
# Load HDF5 file, set up Dedalus, build LBVP for Poisson solve
# ---------------------------------------------------------------------------

f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

# Detect whether u @ grad(omega) is already saved in the HDF5.
have_nonlin_omega = 'nonlin_omega' in f['tasks']
if rank == 0:
    if have_nonlin_omega:
        logger.info("'nonlin_omega' present in HDF5 -- will load directly.")
    else:
        logger.info("'nonlin_omega' absent -- will compute u @ grad(vort) post-hoc.")

dealias = 3 / 2
dtype   = np.float64
coords  = d3.PolarCoordinates('phi', 'r')
dist    = d3.Distributor(coords, dtype=dtype)
disk    = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                        dealias=dealias, dtype=dtype)
edge         = disk.edge
phi_local, r_local = dist.local_grids(disk)

u    = dist.VectorField(coords, name='u',    bases=disk)
vort = dist.Field(             name='vort', bases=disk)

# Streamfunction and tau fields for the LBVP.
psi      = dist.Field(name='psi',     bases=disk)
tau_psi  = dist.Field(name='tau_psi', bases=edge)
tau_psi2 = dist.Field(name='tau_psi2')

lift = lambda A: d3.Lift(A, disk, -1)

# Build the LBVP once.  Per snapshot we just refresh `vort` and call solve().
problem_psi = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
problem_psi.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vort")
problem_psi.add_equation("psi(r=1) = 0")
problem_psi.add_equation("integ(psi) = 0")
solver_psi = problem_psi.build_solver()

# Scalar field to hold the saved nonlin_omega when present.
if have_nonlin_omega:
    nomega_field = dist.Field(name='nonlin_omega', bases=disk)

# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature weights (from Dedalus's own infrastructure)
# ---------------------------------------------------------------------------

r_1d = r_local[0, :]
z0, w_r = zsphere.quadrature(2, Nr, k=0)
_qs = float(np.sum(w_r))
if not np.isclose(_qs, 0.5, atol=1e-12):
    raise RuntimeError(
        f"Dedalus Gauss-Legendre weights do not sum to 1/2: got {_qs:.16f}")
_r_from_z0 = np.sqrt(0.5 * (1.0 + z0))
if not np.allclose(np.sort(r_1d), np.sort(_r_from_z0), atol=1e-10):
    raise RuntimeError(
        "Dedalus radial grid does not match zernike.quadrature nodes.")

# ---------------------------------------------------------------------------
# Precompute scalar Dirichlet basis matrices:
#   B[m][i, n] = J_m(lambda_{m,n}^D * r_i) * w_r[i]
# Projection of a profile  f(r_i)  onto the n-th Bessel mode:
#   c_n = (1/dirichlet_ws[m][n]) * (f @ B[m])[n]
# ---------------------------------------------------------------------------

B = {}
for m in ms:
    B[m] = (
        sp.jv(m, dirichlet_zs[m][np.newaxis, :] * r_1d[:, np.newaxis])
        * w_r[:, np.newaxis]
    )

# ---------------------------------------------------------------------------
# Binning: shared k-grid using Dirichlet roots of J_0 as bin centres
# (matches Route C convention for direct cross-comparison)
# ---------------------------------------------------------------------------

def define_bins(centers):
    Nbins = len(centers)
    edges = np.array([centers[i] - 0.5 * (centers[i] - centers[i - 1])
                      for i in range(1, Nbins)])
    edges = np.concatenate(([0.0], edges, [centers[-1] + np.pi / 2]))
    return Nbins, edges

bin_centers = sp.jn_zeros(0, Nr)
Nbins, bin_edges = define_bins(bin_centers)
bin_widths = np.diff(bin_edges)

def bin_one_m_resolved(values, data, mmax_local):
    """Bin (mmax+1, Nr) eigenvalues + data into (mmax+1, Nbins)."""
    out = np.zeros((mmax_local + 1, Nbins))
    for m in range(mmax_local + 1):
        for bb in range(Nbins):
            mask = (values[m, :] >= bin_edges[bb]) & (values[m, :] < bin_edges[bb + 1])
            out[m, bb] = np.sum(data[m, mask]) / bin_widths[bb]
    return out

# ---------------------------------------------------------------------------
# Select snapshots to process
# ---------------------------------------------------------------------------

#ws = np.arange(np.where(t <= t_out_start)[0][-1],
#               np.where(t >= t_out_end)[0][0] + 1)
ws = np.arange(np.where(t == np.min(t))[0][0], np.where(t == np.max(t))[0][0] + 1)
nw = len(ws)
tw = t[ws]

# ---------------------------------------------------------------------------
# Allocate output arrays (rank 0)
# ---------------------------------------------------------------------------

# Per (snapshot, m, n)
E_psiB = np.zeros((nw, mmax + 1, Nr))  # E_psi(m, n)
T_psiB = np.zeros((nw, mmax + 1, Nr))  # T_psi(m, n)

# 1D binned
E_psiBn = np.zeros((nw, Nbins))
T_psiBn = np.zeros((nw, Nbins))

# m-resolved binned
E_psiBmn = np.zeros((nw, mmax + 1, Nbins))
T_psiBmn = np.zeros((nw, mmax + 1, Nbins))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

prog_cad = 32

for i, w in enumerate(ws):
    if i % prog_cad == 0 and rank == 0:
        logger.info("Snapshot loop: i = %d / %d  (t = %.3f)", i, nw, tw[i])

    # --- Load u and vort ---
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)

    # --- Solve Poisson for psi ---
    # The LBVP was built with vort as the RHS expression; updating vort's
    # data and re-calling solve() refreshes the right-hand side.
    solver_psi.solve()
    psi.change_scales(1)

    # --- Obtain nonlin_omega: load if available, else compute post-hoc ---
    if have_nonlin_omega:
        nomega_field.load_from_hdf5(f, w)
        nomega_field.change_scales(1)
        nomega_loc = np.copy(nomega_field['g'])
    else:
        nomega_expr = (u @ d3.grad(vort)).evaluate()
        nomega_expr.change_scales(1)
        nomega_loc = np.copy(nomega_expr['g'])

    psi_loc = np.copy(psi['g'])

    # --- Gather grids to rank 0 ---
    psi_g_list  = comm.gather(psi_loc,    root=0)
    nom_g_list  = comm.gather(nomega_loc, root=0)

    if rank != 0:
        continue

    psi_g = np.concatenate(psi_g_list, axis=0)
    nom_g = np.concatenate(nom_g_list, axis=0)

    # --- phi-FFT to (a_m, b_m) amplitudes ---
    a_psi, b_psi = phi_fft_to_cos_sin(psi_g, Nphi)
    a_nom, b_nom = phi_fft_to_cos_sin(nom_g, Nphi)

    # --- Per-m projection and spectra ---
    for m in range(mmax + 1):
        wD = dirichlet_ws[m]
        lam = dirichlet_zs[m]

        # Scalar Dirichlet projection via matrix-vector product.
        c_psi_cos = (a_psi[m, :] @ B[m]) / wD
        c_psi_sin = (b_psi[m, :] @ B[m]) / wD
        c_nom_cos = (a_nom[m, :] @ B[m]) / wD
        c_nom_sin = (b_nom[m, :] @ B[m]) / wD

        # Azimuthal factors:  ∫_0^{2pi} cos^2(m phi) d phi = pi (m>0) or 2 pi (m=0)
        #                     ∫_0^{2pi} sin^2(m phi) d phi = pi (m>0) or 0 (m=0)
        phi_factor_cos = 2 * np.pi if m == 0 else np.pi
        phi_factor_sin = 0.0       if m == 0 else np.pi

        # Energy per mode:
        #   E_psi(m,n) = (1/2) * lambda^2 * |psi|^2  integrated, by Parseval +
        #   the eigenvalue relation  lap(psi) = -lambda^2 psi  (psi in Dirichlet
        #   basis with psi(r=1)=0 by gauge).
        E_psiB[i, m, :] = 0.5 * lam ** 2 * wD * (
            phi_factor_cos * c_psi_cos ** 2 + phi_factor_sin * c_psi_sin ** 2
        )

        # Transfer per mode:  T_psi(m,n) = +<psi, u@grad(omega)>
        T_psiB[i, m, :] = wD * (
            phi_factor_cos * c_psi_cos * c_nom_cos
            + phi_factor_sin * c_psi_sin * c_nom_sin
        )

    # --- Bin per snapshot ---
    E_psiBmn[i, :, :] = bin_one_m_resolved(lambda_arr, E_psiB[i], mmax)
    T_psiBmn[i, :, :] = bin_one_m_resolved(lambda_arr, T_psiB[i], mmax)
    E_psiBn[i, :] = np.sum(E_psiBmn[i], axis=0)
    T_psiBn[i, :] = np.sum(T_psiBmn[i], axis=0)

# ---------------------------------------------------------------------------
# Dissipation, time derivative, budget residual (rank 0 only)
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Computing dissipation spectra and budget residual")

    # D_nu(m, n) = 2*nu * lambda^2 * E_psi(m, n)   -- exact in Dirichlet basis
    # (omega = lap(psi) = -lambda^2 psi per mode; no metric correction needed)
    D_nu_Bmn = np.zeros((nw, Nbins))
    for i in range(nw):
        d_per_mode = 2.0 * nu * lambda_arr ** 2 * E_psiB[i]
        d_binned   = bin_one_m_resolved(lambda_arr, d_per_mode, mmax)
        D_nu_Bmn[i, :] = np.sum(d_binned, axis=0)

    # D_alpha(k, t) = 2*alpha * E(k, t)
    D_alpha_Bn = 2.0 * alpha * E_psiBn

    # dE/dt via central finite differences in time
    if nw > 1:
        dE_psiBn_dt = np.gradient(E_psiBn, tw, axis=0)
    else:
        dE_psiBn_dt = np.zeros_like(E_psiBn)

    # Budget residual:  dE/dt - T + D_nu + D_alpha   (forcing omitted)
    budget_residual_Bn = (
        dE_psiBn_dt - T_psiBn + D_nu_Bmn + D_alpha_Bn
        if nw > 1 else None
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Saving: %s_%s.npy", output_prefix, output_suffix)

    processed = {}
    processed['ws']           = ws
    processed['ts']           = tw
    processed['ms']           = ms
    processed['Nbins']        = Nbins
    processed['bin_centers']  = bin_centers
    processed['bin_edges']    = bin_edges
    processed['lambda_arr']   = lambda_arr
    processed['have_nonlin_omega'] = have_nonlin_omega

    if not steady_only:
        processed['E_psiB']   = E_psiB
        processed['T_psiB']   = T_psiB
        processed['E_psiBn']  = E_psiBn
        processed['T_psiBn']  = T_psiBn
        processed['E_psiBmn'] = E_psiBmn
        processed['T_psiBmn'] = T_psiBmn
        processed['D_nu_Bn']  = D_nu_Bmn
        processed['D_alpha_Bn'] = D_alpha_Bn
        processed['dE_psiBn_dt'] = dE_psiBn_dt
        if budget_residual_Bn is not None:
            processed['budget_residual_Bn'] = budget_residual_Bn

    if nw > 1:
        t_end     = tw[-1]
        t_start   = t_end - t_steady_range
        idx_end   = np.where(tw <= t_end)[0][-1]
        idx_start = np.where(tw >= t_start)[0][0]
        sl = slice(idx_start, idx_end)

        for key, arr in [('E_psiB', E_psiB), ('T_psiB', T_psiB),
                         ('E_psiBn', E_psiBn), ('T_psiBn', T_psiBn),
                         ('E_psiBmn', E_psiBmn), ('T_psiBmn', T_psiBmn),
                         ('D_nu_Bn', D_nu_Bmn), ('D_alpha_Bn', D_alpha_Bn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)
        if budget_residual_Bn is not None:
            processed['budget_residual_Bn_tavg'] = np.mean(
                budget_residual_Bn[sl], axis=0)

    np.save(output_prefix + '_' + output_suffix + '.npy', processed)
    logger.info("Save complete")
else:
    logger.info("Rank %d finished", rank)

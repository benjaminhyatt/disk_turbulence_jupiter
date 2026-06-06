"""
Streamfunction (option G) post-processing for 2D disk turbulence.

Analytical-Bouwkamp version (no runtime quadrature).

For each saved snapshot:
  1. Load u (vector) and vort (scalar = omega).
  2. Solve  lap(psi) = vort  with gauge  psi(r=1) = 0  via LBVP.  This makes
     psi exactly Dirichlet at the boundary by construction.
  3. Obtain  u @ grad(vort)  (= u . grad(omega)) as a scalar field, called
     'nonlin_omega' here:
        * If the HDF5 file contains a saved 'nonlin_omega' task, use it.
        * Otherwise compute it post-hoc via (u @ d3.grad(vort)).evaluate().
  4. Read off the *Zernike* coefficients of psi and nonlin_omega directly
     (psi['c'] and nomega['c']) -- Dedalus has already done the radial
     spectral transform for us.
  5. Apply the analytical Zernike->Bessel transform matrix (zern2std), built
     from the Bouwkamp identity
        int_0^1 r Z_n^m(r) J_m(lambda r) dr = (-1)^p J_{n+1}(lambda)/lambda
     to obtain Bessel coefficients in the scalar Dirichlet J_m basis.  This
     replaces the previous grid-space Gauss-Legendre quadrature with an
     exact analytical projection, eliminating the spurious high-k tail
     caused by quadrature error at high Bessel modes.
  6. Form the spectra
        E_psi(m, n) = (1/2) * lambda_{m,n}^2 * |psi_hat_{m,n}|^2 * (norm)
        T_psi(m, n) = + <psi_hat, (u@grad omega)_hat>_{m,n} * (norm)
     Bin into shared k-grid.  In this gauge,
        D_alpha = 2*alpha*E_psi        (exact)
        D_nu    = 2*nu*lambda^2*E_psi  (exact in this basis, because
                                        omega = lap(psi) = -lambda^2 psi
                                        per Dirichlet mode)
     so the entire budget closes within the scalar Dirichlet basis.

This pipeline mirrors v2's std-branch behaviour for E_psi, with the T_psi
projection now included.  Serves as an independent cross-basis benchmark
for the route-C (process_spectra_zb_mbin_v3.py) pipeline.

Usage:
    process_spectra_option_G.py <file>... [options]

Options:
    --t_out_start=<value>       Simulation time to begin making spectra, or "auto" for the first saved snapshot's time [default: auto]
    --t_out_end=<value>         Simulation time to stop making spectra, or "auto" for the last saved snapshot's time [default: auto]
    --t_steady_range=<value>    Size of time window prior to t_out_end to average as steady state, or "auto" for the full processed duration [default: auto]
    --make_new=<bool>           Rebuild the Zernike-to-Bessel MMT matrices from scratch (otherwise load cached) [default: False]
    --steady_only=<bool>        True: save only time-averaged steady-state data; False: save all timesteps [default: True]
"""

import numpy as np
import h5py
import scipy.special as sp
import dedalus.public as d3
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
# Parameter parsing from output filename (same convention as A / C)
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
nu_str    = output_suffix.split('nu_')[1].split('_')[0]

alpha_vals = np.array([1e-2, 2e-2, 4e-2])
nu_vals    = np.array([1e-4, 2e-4, 4e-4])

alpha   = alpha_vals[np.argmin(np.abs(alpha_vals - str_to_float(alpha_str)))]
nu      = nu_vals[np.argmin(np.abs(nu_vals       - str_to_float(nu_str)))]

t_out_start_arg    = args['--t_out_start']
t_out_end_arg      = args['--t_out_end']
t_steady_range_arg = args['--t_steady_range']
make_new       = eval(args['--make_new'])
steady_only    = eval(args['--steady_only'])

output_prefix = 'processed_spectra_optionG'
output_prefix += '_steady' if steady_only else ''

# ---------------------------------------------------------------------------
# Scalar Dirichlet Bessel + analytical Bouwkamp helpers
# ---------------------------------------------------------------------------

def std_roots(m, N):
    """First N positive roots of J_m(z) = 0."""
    return sp.jn_zeros(m, N)

def std_weights(m, std_zsm):
    """Standard Fourier-Bessel (Dirichlet) normalisation:
       int_0^1 J_m(z r)^2 r dr = (1/2) J_{m+1}(z)^2   for z a root of J_m.
    """
    return 0.5 * sp.jv(m + 1, std_zsm) ** 2

def zern2std(m, Nr, std_zsm):
    """Analytical Zernike-to-Bessel (scalar Dirichlet) transform matrix
    for azimuthal mode m.  Derived from the Bouwkamp identity:
       int_0^1 r Z_n^m(r) J_m(lambda r) dr = (-1)^p J_{n+1}(lambda)/lambda.
    Multiplying Dedalus's Zernike coefficient vector for a scalar field by
    this matrix yields the Bessel coefficients directly -- no quadrature.
    """
    nstart = int(np.floor(m / 2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = ((-1) ** (jj - 1)
                         * np.sqrt(2 * (2 * jj + m - 1))
                         * (2 * sp.jv(2 * jj + m - 1, std_zsm[i]))
                         / (std_zsm[i] * sp.jv(m + 1, std_zsm[i]) ** 2))
    return ZBm

def makeZBs_std(Nr, Nphi, std_zs):
    """Build the per-m Zernike->Dirichlet-Bessel MMT matrices."""
    mmax = int(Nphi / 2) - 1
    ZBs = {}
    for m in range(mmax + 1):
        ZBs[m] = zern2std(m, Nr, std_zs[m, :])
        if m % 32 == 0 and rank == 0:
            logger.info("makeZBs_std: m = %d / %d", m, mmax)
    return ZBs

def m_map(m, Nphi):
    """Map azimuthal mode m to its cos-slot index in Dedalus's disk-basis
    coefficient layout (factor 4 per m).  The sin-slot index is the cos
    index + 1.  For scalar fields the layout uses the same 4-slots-per-m
    structure as the spin-weighted vector layout (with two slots empty per
    m for scalar -- they don't break the indexing).
    """
    m_in  = np.atleast_1d(np.array(m))
    m_out = 4 * m_in
    mask  = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi / 4))
    return m_out

def ke_m_std(m, Nr, psimc, psims, std_zs):
    """Kinetic energy spectrum contribution at azimuthal order m:
       E_psi(m, n) = (1/2) * lambda^2 * |psi|^2  (Parseval + eigenvalue
       relation lap(psi) = -lambda^2 psi at each Dirichlet mode).
    """
    w = std_weights(m, std_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * 0.5 * std_zs[m, :] ** 2 * psimc ** 2
    else:
        return np.pi * w * 0.5 * std_zs[m, :] ** 2 * (psimc ** 2 + psims ** 2)

def t_m_std(m, Nr, psimc, psims, nmc, nms, std_zs):
    """Nonlinear transfer spectrum at azimuthal order m:
    T_psi(m, n) = + <psi, u@grad(omega)>(m, n)
    Sign convention from the streamfunction-formulation budget derivation:
       dE/dt = -int psi * omega_t dA, and substituting omega_t = -u@grad(omega) + ...
       gives a POSITIVE  +int psi * (u@grad(omega)) dA  in the transfer slot.
    No factor of 1/2 here (unlike E, which carries it from KE = (1/2)|u|^2).
    """
    w = std_weights(m, std_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * psimc * nmc
    else:
        return np.pi * w * (psimc * nmc + psims * nms)

def en_m_std(m, Nr, vortmc, vortms, std_zs):
    """Enstrophy spectrum contribution: En(m, n) = lambda^4 * |psi|^2,
    using omega = lap(psi) = -lambda^2 psi at each Dirichlet mode.
    """
    w = std_weights(m, std_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * std_zs[m, :] ** 4 * vortmc ** 2
    else:
        return np.pi * w * std_zs[m, :] ** 4 * (vortmc ** 2 + vortms ** 2)

# ---------------------------------------------------------------------------
# Binning utilities  (use the same bin convention as Routes A and C)
# ---------------------------------------------------------------------------

def define_bins(zs, centers):
    Nbins = len(centers)
    edges = np.array([centers[ii] - 0.5 * (centers[ii] - centers[ii - 1])
                      for ii in range(1, Nbins)])
    edges = np.concatenate(([0.], edges, [centers[-1] + np.pi / 2]))
    counts, masks = [], {}
    for b in range(Nbins):
        mask        = (zs >= edges[b]) & (zs <= edges[b + 1])
        counts.append(int(np.sum(mask)))
        masks[b]    = mask
    return Nbins, edges, counts, masks

def define_bins_m(zs, mmax, m_keep, centers, edges):
    Nbins = len(centers)
    counts, masks = [], {}
    for b in range(Nbins):
        mask = (zs >= edges[b]) & (zs <= edges[b + 1])
        for m in range(mmax + 1):
            if m != m_keep:
                mask[m, :] = False
        counts.append(int(np.sum(mask)))
        masks[b] = mask
    return Nbins, counts, masks

def bin_spectra(data, widths, Nbins, masks):
    return [np.sum(data[masks[b]] / widths[b]) for b in range(Nbins)]

# ---------------------------------------------------------------------------
# Compute Dirichlet roots and build ZB matrices (cached to disk)
# ---------------------------------------------------------------------------

mmax = int(Nphi / 2) - 1
ms   = np.arange(mmax + 1)

std_zs = np.zeros((mmax + 1, Nr))
for m in range(mmax + 1):
    std_zs[m, :] = std_roots(m, Nr)

filedir  = f'mmts_std_Nr_{Nr}_Nphi_{Nphi}/'
filename = f'zb_std_Nr_{Nr}_Nphi_{Nphi}.npy'

if not make_new and Path(filedir + filename).exists():
    ZBs = np.load(filedir + filename, allow_pickle=True)[()]
    if rank == 0:
        logger.info("ZBs loaded from %s", filedir + filename)
else:
    if rank == 0:
        logger.info("Building ZBs (zern2std) matrices from scratch")
    ZBs = makeZBs_std(Nr, Nphi, std_zs)
    if rank == 0:
        Path(filedir).mkdir(parents=True, exist_ok=True)
        np.save(filedir + filename, ZBs, allow_pickle=True)
if rank == 0:
    logger.info("ZBs ready")

# ---------------------------------------------------------------------------
# Load HDF5 and set up Dedalus
# ---------------------------------------------------------------------------

f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

# Resolve "auto" defaults now that we know the snapshot time range.
t_out_start    = float(t[0])  if str(t_out_start_arg).lower() == 'auto' else float(t_out_start_arg)
t_out_end      = float(t[-1]) if str(t_out_end_arg).lower()   == 'auto' else float(t_out_end_arg)
t_steady_range = (t_out_end - t_out_start) if str(t_steady_range_arg).lower() == 'auto' else float(t_steady_range_arg)
if rank == 0:
    logger.info("Time window: [%.4f, %.4f], steady_range=%.4f",
                t_out_start, t_out_end, t_steady_range)

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

u    = dist.VectorField(coords, name='u',    bases=disk)
vort = dist.Field(             name='vort', bases=disk)

# Streamfunction and tau fields for the LBVP.
psi      = dist.Field(name='psi',     bases=disk)
tau_psi  = dist.Field(name='tau_psi', bases=edge)
tau_psi2 = dist.Field(name='tau_psi2')

lift = lambda A: d3.Lift(A, disk, -1)

problem_psi = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
problem_psi.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vort")
problem_psi.add_equation("psi(r=1) = 0")
problem_psi.add_equation("integ(psi) = 0")
solver_psi = problem_psi.build_solver()

if have_nonlin_omega:
    nomega_field = dist.Field(name='nonlin_omega', bases=disk)

# ---------------------------------------------------------------------------
# Binning setup -- shared k-grid with Routes A and C (J_0 Dirichlet roots)
# ---------------------------------------------------------------------------

bin_centers = sp.jn_zeros(0, Nr)
Nbins, bin_edges, _, masks = define_bins(std_zs, bin_centers)
bin_widths = np.diff(bin_edges)

masks_m = {}
for m in range(mmax + 1):
    _, _, mask_m = define_bins_m(std_zs, mmax, m, bin_centers, bin_edges)
    masks_m[m] = mask_m

# ---------------------------------------------------------------------------
# Select snapshots to process
# ---------------------------------------------------------------------------

ws = np.arange(np.where(t <= t_out_start)[0][-1],
               np.where(t >= t_out_end)[0][0] + 1)
nw = len(ws)
tw = t[ws]

# ---------------------------------------------------------------------------
# Allocate output arrays (rank 0)
# ---------------------------------------------------------------------------

# Coefficient stores
psiZ    = np.zeros((nw, Nphi, Nr))
psiB    = np.zeros((nw, Nphi, Nr))
nomZ    = np.zeros((nw, Nphi, Nr))
nomB    = np.zeros((nw, Nphi, Nr))
vortZ   = np.zeros((nw, Nphi, Nr))
vortB   = np.zeros((nw, Nphi, Nr))

# 2D spectra
keB = np.zeros((nw, mmax + 1, Nr))
enB = np.zeros((nw, mmax + 1, Nr))
tB  = np.zeros((nw, mmax + 1, Nr))

# 1D binned spectra
keBn = np.zeros((nw, Nbins))
enBn = np.zeros((nw, Nbins))
tBn  = np.zeros((nw, Nbins))

# m-resolved binned spectra
keBmn = np.zeros((nw, mmax + 1, Nbins))
enBmn = np.zeros((nw, mmax + 1, Nbins))
tBmn  = np.zeros((nw, mmax + 1, Nbins))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

prog_cad = 32

for i, w in enumerate(ws):
    if i % prog_cad == 0 and rank == 0:
        logger.info("Loop: i = %d / %d  (t = %.3f)", i, nw, tw[i])

    # --- Load u, vort ---
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)

    # --- Solve Poisson for psi ---
    solver_psi.solve()
    psi.change_scales(1)

    # --- Compute / load nonlin_omega ---
    if have_nonlin_omega:
        nomega_field.load_from_hdf5(f, w)
        nomega_field.change_scales(1)
        nomega_obj = nomega_field
    else:
        nomega_obj = (u @ d3.grad(vort)).evaluate()
        nomega_obj.change_scales(1)

    # --- Gather Zernike coefficients to rank 0 ---
    vort.change_scales(1)

    psiZ_loc  = np.copy(psi['c'])
    nomZ_loc  = np.copy(nomega_obj['c'])
    vortZ_loc = np.copy(vort['c'])

    psiZ_list  = comm.gather(psiZ_loc,  root=0)
    nomZ_list  = comm.gather(nomZ_loc,  root=0)
    vortZ_list = comm.gather(vortZ_loc, root=0)

    if rank != 0:
        continue

    psiZ[i, :, :]  = np.array(psiZ_list ).reshape(Nphi, Nr)
    nomZ[i, :, :]  = np.array(nomZ_list ).reshape(Nphi, Nr)
    vortZ[i, :, :] = np.array(vortZ_list).reshape(Nphi, Nr)

    # --- Apply analytical Bouwkamp MMT per azimuthal mode m ---
    for m in range(mmax + 1):
        ZB   = ZBs[m]
        midx = m_map(m, Nphi)
        mc, ms_idx = midx, midx + 1

        psiB[i, mc, :]     = (ZB @ psiZ[i, mc, :][0, :]).reshape(1, Nr)
        psiB[i, ms_idx, :] = (ZB @ psiZ[i, ms_idx, :][0, :]).reshape(1, Nr)

        nomB[i, mc, :]     = (ZB @ nomZ[i, mc, :][0, :]).reshape(1, Nr)
        nomB[i, ms_idx, :] = (ZB @ nomZ[i, ms_idx, :][0, :]).reshape(1, Nr)

        vortB[i, mc, :]     = (ZB @ vortZ[i, mc, :][0, :]).reshape(1, Nr)
        vortB[i, ms_idx, :] = (ZB @ vortZ[i, ms_idx, :][0, :]).reshape(1, Nr)

        # Per-mode spectra
        keB[i, m, :] = ke_m_std(m, Nr,
                                 psiB[i, mc, :], psiB[i, ms_idx, :], std_zs)
        tB[i, m, :]  = t_m_std (m, Nr,
                                 psiB[i, mc, :], psiB[i, ms_idx, :],
                                 nomB[i, mc, :], nomB[i, ms_idx, :], std_zs)
        # Enstrophy: use vorticity coefficients directly
        enB[i, m, :] = en_m_std(m, Nr,
                                 vortB[i, mc, :], vortB[i, ms_idx, :], std_zs) / std_zs[m, :] ** 4
        # ^^ This division strips the lambda^4 factor that en_m_std puts on
        # psi-based enstrophy.  The reason: we're using vortB directly here
        # (NOT psi-derived), so we want En(m,n) = (1/2) <vort, vort>(m,n) =
        # 0.5 * w * (vortmc^2 + vortms^2) * pi_factor.  The en_m_std function
        # was written for psi-derived (omega = lambda^2 psi).  Simpler: use a
        # direct formula here.
        if m == 0:
            enB[i, m, :] = 2 * np.pi * std_weights(m, std_zs[m, :]) * vortB[i, mc, :] ** 2
        else:
            enB[i, m, :] = np.pi * std_weights(m, std_zs[m, :]) * (
                vortB[i, mc, :] ** 2 + vortB[i, ms_idx, :] ** 2
            )

    # --- Bin per snapshot ---
    for m in range(mmax + 1):
        keBmn[i, m, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks_m[m])
        enBmn[i, m, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks_m[m])
        tBmn[i, m, :]  = bin_spectra(tB[i, :, :],  bin_widths, Nbins, masks_m[m])

    keBn[i, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks)
    enBn[i, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks)
    tBn[i, :]  = bin_spectra(tB[i, :, :],  bin_widths, Nbins, masks)

# ---------------------------------------------------------------------------
# Dissipation, time derivative, budget residual (rank 0 only)
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Computing dissipation spectra and budget residual")

    # D_nu(m, n) = 2*nu * lambda^2 * E_psi(m, n)   -- exact in this basis
    # (omega = lap(psi) = -lambda^2 psi gives this without metric corrections)
    D_nu_Bn = np.zeros((nw, Nbins))
    for i in range(nw):
        d_per_mode = 2.0 * nu * std_zs ** 2 * keB[i]   # shape (mmax+1, Nr)
        for b in range(Nbins):
            D_nu_Bn[i, b] = np.sum(d_per_mode[masks[b]] / bin_widths[b])

    # D_alpha = 2*alpha * E
    D_alpha_Bn = 2.0 * alpha * keBn

    if nw > 1:
        dkeBn_dt = np.gradient(keBn, tw, axis=0)
    else:
        dkeBn_dt = np.zeros_like(keBn)

    # Budget residual:  dE/dt - T + D_nu + D_alpha   (forcing omitted)
    budget_residual_Bn = (
        dkeBn_dt - tBn + D_nu_Bn + D_alpha_Bn
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
    processed['std_zs']       = std_zs
    processed['have_nonlin_omega'] = have_nonlin_omega

    if not steady_only:
        processed['psiZ']  = psiZ
        processed['psiB']  = psiB
        processed['nomZ']  = nomZ
        processed['nomB']  = nomB
        processed['vortZ'] = vortZ
        processed['vortB'] = vortB

        processed['keB']   = keB
        processed['enB']   = enB
        processed['tB']    = tB

        processed['keBn']      = keBn
        processed['enBn']      = enBn
        processed['tBn']       = tBn
        processed['D_nu_Bn']   = D_nu_Bn
        processed['D_alpha_Bn'] = D_alpha_Bn
        processed['dkeBn_dt']  = dkeBn_dt
        if budget_residual_Bn is not None:
            processed['budget_residual_Bn'] = budget_residual_Bn

        processed['keBmn'] = keBmn
        processed['enBmn'] = enBmn
        processed['tBmn']  = tBmn

    if nw > 1:
        t_end     = tw[-1]
        t_start   = t_end - t_steady_range
        idx_end   = np.where(tw <= t_end)[0][-1]
        idx_start = np.where(tw >= t_start)[0][0]
        sl = slice(idx_start, idx_end)

        for key, arr in [('keB', keB), ('enB', enB), ('tB', tB)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        for key, arr in [('keBn', keBn), ('enBn', enBn), ('tBn', tBn),
                         ('D_nu_Bn', D_nu_Bn), ('D_alpha_Bn', D_alpha_Bn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        for key, arr in [('keBmn', keBmn), ('enBmn', enBmn), ('tBmn', tBmn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        if budget_residual_Bn is not None:
            processed['budget_residual_Bn_tavg'] = np.mean(
                budget_residual_Bn[sl], axis=0)

    np.save(output_prefix + '_' + output_suffix + '.npy', processed)
    logger.info("Save complete")
else:
    logger.info("Rank %d finished", rank)

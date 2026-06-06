"""
Option A post-processing for 2D disk turbulence.

This is the v2-style pipeline, isolated to a single-purpose script:
  * Spin+/- coefficient representation kept throughout (no de-spinning).
  * Bessel basis index matches azimuthal Fourier mode: J_m (scalar regularity),
    not J_{|m-1|}.
  * H = +1 Dini (Robin) boundary condition for the radial Bessel system.
    This is the basis used in the original v2 processing.  H=+1 does not
    physically match either impenetrability or stress-free, but it is the
    L^2-complete system v2 was tuned to and produces consistent E, T, etc.
  * Zernike -> Bessel transform uses the analytical Bouwkamp identity via
    the `zern2dini` matrices (precomputed once, cached to disk).
  * Plancherel inner products in (u_+, u_-): total E and T come out as
    diagonal coefficient sums; no E_r vs E_phi split.

Per snapshot:
  1. Load u (vector), vort (scalar), and nonlin = u@grad(u) (vector).
  2. Extract spin+ and spin- Zernike coefficient arrays.
  3. For each azimuthal m, apply the precomputed Zernike->Bessel MMT
     (one matmul per spin component per slot) to obtain Bessel coefficients.
  4. Form E(m, n), T(m, n), En(m, n) using the spin± inner products.
  5. Bin into shared k-grid (Dirichlet roots of J_0 as bin centres, matching
     Routes C and G for direct cross-comparison).
  6. Post-loop: D_alpha = 2*alpha*E and D_nu_approx = 2*nu*lambda^2*E (the
     standard approximation; vector-Laplacian corrections not captured).

Provided as the "baseline" pipeline against which Routes C and G are
compared.  Numerically should be the same as the v2 dini-branch processing,
just with a cleaner standalone code organisation.

Usage:
    process_spectra_option_A.py <file>... [options]

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
from scipy.optimize import newton
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
# Parameter parsing from output filename (same convention as v2/v3/G)
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

t_out_start_arg    = args['--t_out_start']
t_out_end_arg      = args['--t_out_end']
t_steady_range_arg = args['--t_steady_range']
make_new       = eval(args['--make_new'])
steady_only    = eval(args['--steady_only'])

output_prefix = 'processed_spectra_optionA'
output_prefix += '_steady' if steady_only else ''

# ---------------------------------------------------------------------------
# Dini (Robin H=+1) helper functions
# ---------------------------------------------------------------------------

H = 1  # Robin parameter for Dini expansion (v2 convention)

def robin_func(r, m, H):
    return np.real(r * sp.jvp(m, r, n=1) + H * sp.jv(m, r))

def robin_func_prime(r, m, H):
    return np.real(r * sp.jvp(m, r, n=2) + (H + 1) * sp.jvp(m, r, n=1))

def dini_roots(m, Nr, H):
    """First Nr positive roots of  z J_m'(z) + H J_m(z) = 0  for H = +1.
    Uses Newton's method seeded from jnp_zeros, as in v2.
    """
    jmp_zs  = sp.jnp_zeros(m, Nr + 1)
    djmp_zs = np.diff(jmp_zs)
    r0 = sp.jnp_zeros(m, 1) if m >= 1 else [1.0]
    roots = []
    for nidx in range(Nr):
        rout, _ = newton(robin_func, r0, fprime=robin_func_prime,
                         args=(m, H), tol=1e-10, full_output=True)
        if rout[0] <= 0:
            raise ValueError("Encountered a negative Dini root")
        if rout[0] in np.unique(roots):
            logger.warning("Duplicate Dini root at m=%d, n=%d", m, nidx)
        if np.abs(robin_func(rout[0], m, H)) > 1e-10:
            logger.warning("Root check failed: |f(root)| = %e",
                           np.abs(robin_func(rout[0], m, H)))
        roots.append(rout[0])
        r0 = rout + djmp_zs[nidx]
    return roots

def dini_weights_direct(m, dini_zsm):
    """Dini normalisation:
       int_0^1 J_m(z r)^2 r dr = (H^2 + z^2 - m^2) J_m(z)^2 / (2 z^2)
       for z a root of  z J_m'(z) + H J_m(z) = 0.
    """
    return ((H ** 2 + dini_zsm ** 2 - m ** 2) * sp.jv(m, dini_zsm) ** 2) / (2 * dini_zsm ** 2)

def zern2dini(m, Nr, dini_zsm):
    """Analytical Zernike-to-Bessel (Dini, H=+1) transform matrix for
    azimuthal mode m.  Uses the Bouwkamp identity:
       int_0^1 r Z_n^m(r) J_m(lambda r) dr  =  (-1)^p J_{n+1}(lambda)/lambda
    where Z_n^m is the Zernike radial polynomial and n - m = 2 p >= 0.
    """
    nstart = int(np.floor(m / 2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = ((-1) ** (jj - 1)
                         * np.sqrt(2 * (2 * jj + m - 1))
                         * (2 * dini_zsm[i] * sp.jv(2 * jj + m - 1, dini_zsm[i]))
                         / ((H ** 2 + dini_zsm[i] ** 2 - m ** 2)
                            * sp.jv(m, dini_zsm[i]) ** 2))
    return ZBm

def makeZBs_dini(Nr, Nphi, dini_zs):
    """Build the per-m Zernike->Dini-Bessel transform matrices."""
    mmax = int(Nphi / 2) - 1
    ZBs = {}
    for m in range(mmax + 1):
        ZBs[m] = zern2dini(m, Nr, dini_zs[m, :])
        if m % 10 == 0 and rank == 0:
            logger.info("makeZBs_dini: m = %d / %d", m, mmax)
    return ZBs

# ---------------------------------------------------------------------------
# Spectral inner products in (u_+, u_-) basis
# ---------------------------------------------------------------------------

def inner_product_m_dini(m, Nr,
                          apmc1, apms1, ammc1, amms1,
                          apmc2, apms2, ammc2, amms2,
                          dini_zs):
    """
    Dini-weighted L^2 inner product of two real vector fields in the spin+/-
    Bessel basis.  Because the fields are real, the cos and sin slots are real
    numbers and the inner product is a plain dot product.

    For kinetic energy:  pass the same (u_+, u_-) twice.
    For transfer T:      pass (u_+, u_-) and (N_+, N_-) where N = u@grad(u),
                         and negate the result (T = -<u, N>).
    """
    w = dini_weights_direct(m, dini_zs[m, :])
    if m == 0:
        # For m=0, nonzero Zernike coefficients live in the sine index slots
        # (Dedalus convention).
        return 2 * np.pi * w * 0.5 * (apms1 * apms2 + amms1 * amms2)
    else:
        return np.pi * w * 0.5 * (apmc1 * apmc2 + apms1 * apms2
                                  + ammc1 * ammc2 + amms1 * amms2)

def ke_m_dini(m, Nr, upmc, upms, ummc, umms, dini_zs):
    """Kinetic energy spectrum contribution from azimuthal order m."""
    return inner_product_m_dini(m, Nr,
                                upmc, upms, ummc, umms,
                                upmc, upms, ummc, umms,
                                dini_zs)

def t_m_dini(m, Nr, upmc, upms, ummc, umms,
                     npmc, npms, nmmc, nmms, dini_zs):
    """Nonlinear energy transfer T(k) at azimuthal order m.
    T(k) = -<u_hat, N_hat>, with N = u@grad(u).
    """
    return -inner_product_m_dini(m, Nr,
                                 upmc, upms, ummc, umms,
                                 npmc, npms, nmmc, nmms,
                                 dini_zs)

def en_m_dini(m, Nr, vortmc, vortms, dini_zs):
    """Enstrophy spectrum contribution from azimuthal order m."""
    w = dini_weights_direct(m, dini_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * vortmc ** 2
    else:
        return np.pi * w * (vortmc ** 2 + vortms ** 2)

# ---------------------------------------------------------------------------
# Azimuthal index mapping (Zernike layout -> m ordering)
# ---------------------------------------------------------------------------

def m_map(m, Nphi):
    """Map azimuthal mode m to its cos-slot index in Dedalus's spin-weighted
    Zernike coefficient layout.  The sin-slot index is one greater.
    """
    m_in  = np.atleast_1d(np.array(m))
    m_out = 4 * m_in
    mask  = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi / 4))
    return m_out

# ---------------------------------------------------------------------------
# Binning utilities (use m=0 Dini roots as bin centres internally, but
# bin_centers below are J_0 Dirichlet roots to match Routes C and G)
# ---------------------------------------------------------------------------

def define_bins(zs, centers):
    """1D wavenumber bins with the supplied centre array."""
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
    """Same binning restricted to a single azimuthal order m_keep."""
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
    """Sum data over each wavenumber bin, normalised by bin width."""
    return [np.sum(data[masks[b]] / widths[b]) for b in range(Nbins)]

# ---------------------------------------------------------------------------
# Flux estimate (Boffetta & Ecke 2012; valid in the inverse cascade range)
# ---------------------------------------------------------------------------

def lambda_k(ke_k, ks):
    return np.sqrt(np.cumsum(ks ** 2 * ke_k))

def flux_k(ke_k, lam_k, ks):
    return np.pi ** (-3 / 2) * lam_k * ks * ke_k

# ---------------------------------------------------------------------------
# Set up Bessel root tables and ZB matrices (cache to disk)
# ---------------------------------------------------------------------------

mmax = int(Nphi / 2) - 1
ms   = np.arange(mmax + 1)

dini_zs = np.zeros((mmax + 1, Nr))
for m in range(mmax + 1):
    dini_zs[m, :] = dini_roots(m, Nr, H)

filedir  = f'mmts_dini_Nr_{Nr}_Nphi_{Nphi}/'
filename = f'zb_dini_Nr_{Nr}_Nphi_{Nphi}.npy'

if not make_new and Path(filedir + filename).exists():
    ZBs = np.load(filedir + filename, allow_pickle=True)[()]
    if rank == 0:
        logger.info("ZBs loaded from %s", filedir + filename)
else:
    if rank == 0:
        logger.info("Building ZBs matrices from scratch")
    ZBs = makeZBs_dini(Nr, Nphi, dini_zs)
    if rank == 0:
        Path(filedir).mkdir(parents=True, exist_ok=True)
        np.save(filedir + filename, ZBs, allow_pickle=True)
if rank == 0:
    logger.info("ZBs ready")

# ---------------------------------------------------------------------------
# Load HDF5 file and set up Dedalus
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

dealias = 3 / 2
dtype   = np.float64
coords  = d3.PolarCoordinates('phi', 'r')
dist    = d3.Distributor(coords, dtype=dtype)
disk    = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                        dealias=dealias, dtype=dtype)
edge         = disk.edge
radial_basis = disk.radial_basis
phi_local, r_local = dist.local_grids(disk)

u            = dist.VectorField(coords, name='u',      bases=disk)
vort         = dist.Field(             name='vort',    bases=disk)

if 'nonlin_u' in list(f['tasks']):
    nonlin_name = 'nonlin_u'
else:
    nonlin_name = 'nonlin'
nonlin_field = dist.VectorField(coords, name=nonlin_name,  bases=disk)

# ---------------------------------------------------------------------------
# Binning setup
# ---------------------------------------------------------------------------

# Use Dirichlet roots of J_0 as bin centres, matching Routes C and G for
# direct cross-comparison.  The mode-by-mode wavenumbers (Dini roots) get
# binned against these centres.
bin_centers = sp.jn_zeros(0, Nr)
Nbins, bin_edges, _, masks = define_bins(dini_zs, bin_centers)
bin_widths = np.diff(bin_edges)
masks_m = {}
for m in range(mmax + 1):
    _, _, mask_m = define_bins_m(dini_zs, mmax, m, bin_centers, bin_edges)
    masks_m[m] = mask_m

# ---------------------------------------------------------------------------
# Select writes to process
# ---------------------------------------------------------------------------

#ws = np.arange(np.where(t <= t_out_start)[0][-1],
#               np.where(t >= t_out_end)[0][0] + 1)
ws = np.arange(np.where(t == np.min(t))[0][0], np.where(t == np.max(t))[0][0] + 1)
nw = len(ws)
tw = t[ws]

# ---------------------------------------------------------------------------
# Allocate arrays (rank 0 keeps the full result)
# ---------------------------------------------------------------------------

# Zernike coefficient stores
vortZ = np.zeros((nw, Nphi, Nr))
upZ   = np.zeros((nw, Nphi, Nr))
umZ   = np.zeros((nw, Nphi, Nr))
npZ   = np.zeros((nw, Nphi, Nr))
nmZ   = np.zeros((nw, Nphi, Nr))

# Bessel coefficient stores
vortB = np.zeros((nw, Nphi, Nr))
upB   = np.zeros((nw, Nphi, Nr))
umB   = np.zeros((nw, Nphi, Nr))
npB   = np.zeros((nw, Nphi, Nr))
nmB   = np.zeros((nw, Nphi, Nr))

# 2D (m, n) spectral arrays
keB = np.zeros((nw, mmax + 1, Nr))
enB = np.zeros((nw, mmax + 1, Nr))
tB  = np.zeros((nw, mmax + 1, Nr))

# 1D binned spectra
keBn   = np.zeros((nw, Nbins))
enBn   = np.zeros((nw, Nbins))
tBn    = np.zeros((nw, Nbins))
fluxBn = np.zeros((nw, Nbins))

# m-resolved binned spectra
keBmn = np.zeros((nw, mmax + 1, Nbins))
enBmn = np.zeros((nw, mmax + 1, Nbins))
tBmn  = np.zeros((nw, mmax + 1, Nbins))

# ---------------------------------------------------------------------------
# Main loop over snapshots
# ---------------------------------------------------------------------------

prog_cad = 32

for i, w in enumerate(ws):
    if i % prog_cad == 0 and rank == 0:
        logger.info("Write loop: i = %d / %d  (t = %.3f)", i, nw, tw[i])

    # --- Load fields ---
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)
    nonlin_field.load_from_hdf5(f, w)

    # --- Gather Zernike coefficients ---
    vort.change_scales(1)
    vortZgather = comm.gather(np.copy(vort['c']), root=0)
    if rank == 0:
        vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)

    u.change_scales(1)
    upZgather = comm.gather(np.copy(u['c'][0, :, :]), root=0)
    umZgather = comm.gather(np.copy(u['c'][1, :, :]), root=0)
    if rank == 0:
        upZ[i, :, :] = np.array(upZgather).reshape(Nphi, Nr)
        umZ[i, :, :] = np.array(umZgather).reshape(Nphi, Nr)

    nonlin_field.change_scales(1)
    npZgather = comm.gather(np.copy(nonlin_field['c'][0, :, :]), root=0)
    nmZgather = comm.gather(np.copy(nonlin_field['c'][1, :, :]), root=0)
    if rank == 0:
        npZ[i, :, :] = np.array(npZgather).reshape(Nphi, Nr)
        nmZ[i, :, :] = np.array(nmZgather).reshape(Nphi, Nr)

    # --- Zernike -> Bessel and spectra (rank 0 only) ---
    if rank == 0:
        for m in range(mmax + 1):
            if m % prog_cad == 0:
                logger.info("ZB transform: m = %d / %d", m, mmax)
            ZB   = ZBs[m]
            midx = m_map(m, Nphi)
            mc, ms_idx = midx, midx + 1

            vortB[i, mc, :]     = (ZB @ vortZ[i, mc, :][0, :]).reshape(1, Nr)
            vortB[i, ms_idx, :] = (ZB @ vortZ[i, ms_idx, :][0, :]).reshape(1, Nr)

            upB[i, mc, :]     = (ZB @ upZ[i, mc, :][0, :]).reshape(1, Nr)
            upB[i, ms_idx, :] = (ZB @ upZ[i, ms_idx, :][0, :]).reshape(1, Nr)
            umB[i, mc, :]     = (ZB @ umZ[i, mc, :][0, :]).reshape(1, Nr)
            umB[i, ms_idx, :] = (ZB @ umZ[i, ms_idx, :][0, :]).reshape(1, Nr)

            npB[i, mc, :]     = (ZB @ npZ[i, mc, :][0, :]).reshape(1, Nr)
            npB[i, ms_idx, :] = (ZB @ npZ[i, ms_idx, :][0, :]).reshape(1, Nr)
            nmB[i, mc, :]     = (ZB @ nmZ[i, mc, :][0, :]).reshape(1, Nr)
            nmB[i, ms_idx, :] = (ZB @ nmZ[i, ms_idx, :][0, :]).reshape(1, Nr)

            keB[i, m, :] = ke_m_dini(m, Nr,
                                      upB[i, mc, :], upB[i, ms_idx, :],
                                      umB[i, mc, :], umB[i, ms_idx, :],
                                      dini_zs)
            enB[i, m, :] = en_m_dini(m, Nr,
                                      vortB[i, mc, :], vortB[i, ms_idx, :],
                                      dini_zs)
            tB[i, m, :]  = t_m_dini( m, Nr,
                                      upB[i, mc, :],  upB[i, ms_idx, :],
                                      umB[i, mc, :],  umB[i, ms_idx, :],
                                      npB[i, mc, :],  npB[i, ms_idx, :],
                                      nmB[i, mc, :],  nmB[i, ms_idx, :],
                                      dini_zs)

        # --- Bin spectra over m ---
        for m in range(mmax + 1):
            keBmn[i, m, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks_m[m])
            enBmn[i, m, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks_m[m])
            tBmn[i, m, :]  = bin_spectra(tB[i, :, :],  bin_widths, Nbins, masks_m[m])

        keBn[i, :]   = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks)
        enBn[i, :]   = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks)
        tBn[i, :]    = bin_spectra(tB[i, :, :],  bin_widths, Nbins, masks)
        fluxBn[i, :] = flux_k(keBn[i, :], lambda_k(keBn[i, :], bin_centers), bin_centers)

# ---------------------------------------------------------------------------
# Dissipation, time derivative, budget residual (rank 0 only)
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Computing dissipation spectra and budget residual")

    # D_nu(m, n) = 2*nu * lambda^2 * E(m, n)   -- scalar-Laplacian approx.
    # (vector Laplacian metric/cross-coupling corrections not captured)
    D_nu_Bn = np.zeros((nw, Nbins))
    for i in range(nw):
        if i % prog_cad == 0 and rank == 0:
            logger.info("Loop: i = %d / %d  (t = %.3f)", i, nw, tw[i])
        #d_per_mode = 2.0 * nu * dini_zs[None, :, :] ** 2 * keB[i]
        d_per_mode = 2.0 * nu * dini_zs[:, :] ** 2 * keB[i]
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
    processed['dini_zs']      = dini_zs
    processed['H']            = H

    if not steady_only:
        # Spectral coefficient stores
        processed['vortZ'] = vortZ
        processed['upZ']   = upZ
        processed['umZ']   = umZ
        processed['npZ']   = npZ
        processed['nmZ']   = nmZ
        processed['vortB'] = vortB
        processed['upB']   = upB
        processed['umB']   = umB
        processed['npB']   = npB
        processed['nmB']   = nmB

        # 2D spectra
        processed['keB']   = keB
        processed['enB']   = enB
        processed['tB']    = tB

        # 1D binned
        processed['keBn']  = keBn
        processed['enBn']  = enBn
        processed['tBn']   = tBn
        processed['fluxBn'] = fluxBn
        processed['D_nu_Bn']    = D_nu_Bn
        processed['D_alpha_Bn'] = D_alpha_Bn
        processed['dkeBn_dt']   = dkeBn_dt
        if budget_residual_Bn is not None:
            processed['budget_residual_Bn'] = budget_residual_Bn

        # m-resolved binned spectra
        processed['keBmn'] = keBmn
        processed['enBmn'] = enBmn
        processed['tBmn']  = tBmn

    # Time-averaged steady-state versions
    if nw > 1:
        t_end     = tw[-1]
        t_start   = t_end - t_steady_range
        idx_end   = np.where(tw <= t_end)[0][-1]
        idx_start = np.where(tw >= t_start)[0][0]
        sl = slice(idx_start, idx_end)

        for key, arr in [('keB', keB), ('enB', enB), ('tB', tB)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)
        processed['keBsum_tavg'] = np.sum(processed['keB_tavg'], axis=1)
        processed['enBsum_tavg'] = np.sum(processed['enB_tavg'], axis=1)

        for key, arr in [('keBn', keBn), ('enBn', enBn), ('tBn', tBn),
                         ('fluxBn', fluxBn),
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

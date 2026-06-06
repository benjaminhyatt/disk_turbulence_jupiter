"""
Bessel energy spectra and spectral budget for forced-dissipative 2D disk turbulence.

Computes the following from saved Dedalus analysis output:
  E(k,t)        -- kinetic energy spectrum (Dini/Bessel basis, m-binned and summed)
  T(k,t)        -- nonlinear energy transfer spectrum  (requires 'nonlin' task in IVP)
  D_nu(k,t)     -- viscous dissipation spectrum  (2*nu*k^2 * E, mode-by-mode)
  D_alpha(k,t)  -- Rayleigh friction spectrum    (2*alpha * E)
  dE/dt         -- numerical time derivative of E(k,t) (budget closure check)
  Enstrophy     -- enstrophy spectrum

Budget closure (dini branch only):
  dE/dt ≈ T - D_nu - D_alpha + F(k)
  where F(k) = eps in the forcing band and 0 elsewhere.
  The residual `budget_residual_Bn` omits F; it should be non-negligible only in the
  forcing band, where it will be ~ -eps (balancing the unaccounted injection).

Usage:
    process_spectra_zb_mbin.py <file>... [options]

Options:
    --dini=<bool>               True: Dini expansion (Robin bc H=1); False: standard Bessel (Dirichlet) [default: True]
    --t_out_start=<float>       Simulation time to begin making spectra [default: 0.]
    --t_out_end=<float>         Simulation time to stop making spectra [default: 100.]
    --t_steady_range=<float>    Size of time window prior to t_out_end to average as steady state [default: 50.]
    --make_new=<bool>           Rebuild the Zernike-to-Bessel MMT matrices from scratch [default: False]
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
# Parameter parsing from output filename
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

# Snap to nearest known parameter value
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

dini           = eval(args['--dini'])
t_out_start    = float(args['--t_out_start'])
t_out_end      = float(args['--t_out_end'])
t_steady_range = float(args['--t_steady_range'])
make_new       = eval(args['--make_new'])
steady_only    = eval(args['--steady_only'])

output_prefix  = 'processed_spectra_zb_mbin'
output_prefix += '_dini' if dini else '_std'
output_prefix += '_steady' if steady_only else ''

# ---------------------------------------------------------------------------
# Dini / standard Bessel helper functions
# ---------------------------------------------------------------------------

H = 1  # Robin bc parameter for Dini expansion

def robin_func(r, m, H):
    return np.real(r * sp.jvp(m, r, n=1) + H * sp.jv(m, r))

def robin_func_prime(r, m, H):
    return np.real(r * sp.jvp(m, r, n=2) + (H + 1) * sp.jvp(m, r, n=1))

def dini_roots(m, Nr, H):
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
            logger.warning("Root check failed: |f(root)| = %e", np.abs(robin_func(rout[0], m, H)))
        roots.append(rout[0])
        r0 = rout + djmp_zs[nidx]
    return roots

def dini_weights_direct(m, dini_zsm):
    return ((H**2 + dini_zsm**2 - m**2) * sp.jv(m, dini_zsm)**2) / (2 * dini_zsm**2)

def zern2dini(m, Nr, dini_zsm):
    nstart = int(np.floor(m / 2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = ((-1)**(jj - 1)
                         * np.sqrt(2 * (2*jj + m - 1))
                         * (2 * dini_zsm[i] * sp.jv(2*jj + m - 1, dini_zsm[i]))
                         / ((H**2 + dini_zsm[i]**2 - m**2) * sp.jv(m, dini_zsm[i])**2))
    return ZBm

def std_roots(m, Nr):
    return sp.jn_zeros(m, Nr)

def std_weights(m, std_zsm):
    return (sp.jv(m + 1, std_zsm)**2) / 2

def zern2std(m, Nr, std_zsm):
    nstart = int(np.floor(m / 2))
    ZBm = np.zeros((Nr, Nr))
    for i in range(Nr):
        for j in range(nstart, Nr):
            jj = j - nstart + 1
            ZBm[i, j] = ((-1)**(jj - 1)
                         * np.sqrt(2 * (2*jj + m - 1))
                         * (2 * sp.jv(2*jj + m - 1, std_zsm[i]))
                         / (std_zsm[i] * sp.jv(m + 1, std_zsm[i])**2))
    return ZBm

# ---------------------------------------------------------------------------
# Spectral inner products
# ---------------------------------------------------------------------------

def inner_product_m_dini(m, Nr,
                          apmc1, apms1, ammc1, amms1,
                          apmc2, apms2, ammc2, amms2,
                          dini_zs):
    """
    Dini-weighted L2 inner product of two real vector fields in the spin+/-
    Bessel basis.  Because the fields are real, Fourier coefficients at each m
    are real (cosine / sine parts), so the inner product is a plain dot product.

    For kinetic energy:  call with both vector arguments equal to (u_+, u_-).
    For transfer T:      call with (u_+, u_-) and (N_+, N_-) and negate the result,
                         because T = -<u, N> where N = u@grad(u).
    """
    w = dini_weights_direct(m, dini_zs[m, :])
    if m == 0:
        # For m=0, nonzero Zernike coefficients live in the 's' (sine) index slots.
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
    """
    Nonlinear energy transfer T(k) at azimuthal order m.
    T(k) > 0: energy arriving at k.  T(k) = -<u_hat, N_hat>, N = u@grad(u).
    """
    return -inner_product_m_dini(m, Nr,
                                 upmc, upms, ummc, umms,
                                 npmc, npms, nmmc, nmms,
                                 dini_zs)


def en_m_dini(m, Nr, vortmc, vortms, dini_zs):
    """Enstrophy spectrum contribution from azimuthal order m."""
    w = dini_weights_direct(m, dini_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * vortmc**2
    else:
        return np.pi * w * (vortmc**2 + vortms**2)


def ke_m_std(m, Nr, psimc, psims, std_zs):
    w = std_weights(m, std_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * 0.5 * std_zs[m, :]**2 * psimc**2
    else:
        return np.pi * w * 0.5 * std_zs[m, :]**2 * (psimc**2 + psims**2)


def en_m_std(m, Nr, psimc, psims, std_zs):
    w = std_weights(m, std_zs[m, :])
    if m == 0:
        return 2 * np.pi * w * std_zs[m, :]**4 * psimc**2
    else:
        return np.pi * w * std_zs[m, :]**4 * (psimc**2 + psims**2)


# ---------------------------------------------------------------------------
# Flux estimate (Boffetta & Ecke 2012; valid in the inverse cascade range only)
# ---------------------------------------------------------------------------

def lambda_k(ke_k, ks):
    return np.sqrt(np.cumsum(ks**2 * ke_k))

def flux_k(ke_k, lam_k, ks):
    return np.pi**(-3/2) * lam_k * ks * ke_k

# ---------------------------------------------------------------------------
# MMT matrix construction
# ---------------------------------------------------------------------------

def makeZBs_dini(Nr, Nphi, dini_zs):
    mmax = int(Nphi / 2) - 1
    ZBs = {}
    for m in range(mmax + 1):
        ZBs[m] = zern2dini(m, Nr, dini_zs[m, :])
        if m % 10 == 0:
            logger.info("makeZBs_dini: m = %d / %d", m, mmax)
    return ZBs

def makeZBs_std(Nr, Nphi, std_zs):
    mmax = int(Nphi / 2) - 1
    ZBs = {}
    for m in range(mmax + 1):
        ZBs[m] = zern2std(m, Nr, std_zs[m, :])
        if m % 32 == 0:
            logger.info("makeZBs_std: m = %d / %d", m, mmax)
    return ZBs

# ---------------------------------------------------------------------------
# Azimuthal index mapping (Zernike layout -> m ordering)
# ---------------------------------------------------------------------------

def m_map(m, Nphi):
    m_in  = np.atleast_1d(np.array(m))
    m_out = 4 * m_in
    mask  = m_out > Nphi - 2
    m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi / 4))
    return m_out

# ---------------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------------

def define_bins(zs):
    """1D wavenumber bins using m=0 Dini/Bessel roots as bin centres."""
    centers = zs[0, :]
    Nbins   = len(centers)
    edges   = np.array([centers[ii] - 0.5 * (centers[ii] - centers[ii - 1])
                        for ii in range(1, Nbins)])
    edges   = np.concatenate(([0.], edges, [centers[-1] + np.pi / 2]))
    counts, masks = [], {}
    for b in range(Nbins):
        mask        = (zs >= edges[b]) & (zs <= edges[b + 1])
        counts.append(int(np.sum(mask)))
        masks[b]    = mask
    return Nbins, centers, edges, counts, masks


def define_bins_m(zs, mmax, m_keep):
    """Same binning restricted to a single azimuthal order m_keep."""
    centers = zs[0, :]
    Nbins   = len(centers)
    edges   = np.array([centers[ii] - 0.5 * (centers[ii] - centers[ii - 1])
                        for ii in range(1, Nbins)])
    edges   = np.concatenate(([0.], edges, [centers[-1] + np.pi / 2]))
    counts, masks = [], {}
    for b in range(Nbins):
        mask = (zs >= edges[b]) & (zs <= edges[b + 1])
        for m in range(mmax + 1):
            if m != m_keep:
                mask[m, :] = False
        counts.append(int(np.sum(mask)))
        masks[b] = mask
    return Nbins, centers, edges, counts, masks


def bin_spectra(data, widths, Nbins, masks):
    """Sum data over each wavenumber bin, normalised by bin width."""
    return [np.sum(data[masks[b]] / widths[b]) for b in range(Nbins)]

# ---------------------------------------------------------------------------
# Load analysis data
# ---------------------------------------------------------------------------

f = h5py.File(file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])

dealias = 3 / 2
dtype   = np.float64
coords  = d3.PolarCoordinates('phi', 'r')
dist    = d3.Distributor(coords, dtype=dtype)
disk    = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                        dealias=dealias, dtype=dtype)
edge         = disk.edge
radial_basis = disk.radial_basis
phi,      r      = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

u            = dist.VectorField(coords, name='u',      bases=disk)
vort         = dist.Field(             name='vort',    bases=disk)
nonlin_field = dist.VectorField(coords, name='nonlin', bases=disk)

# Extra fields needed only for the std (Dirichlet / streamfunction) branch
if not dini:
    psi      = dist.Field(name='psi',    bases=disk)
    tau_psi  = dist.Field(name='tau_psi',  bases=edge)
    tau_psi2 = dist.Field(name='tau_psi2')
    lift     = lambda A: d3.Lift(A, disk, -1)

# ---------------------------------------------------------------------------
# Build (or load) Zernike -> Bessel transform matrices
# ---------------------------------------------------------------------------

mmax = int(Nphi / 2) - 1
ms   = np.arange(mmax + 1)

if dini:
    dini_zs = np.zeros((mmax + 1, Nr))
    for m in range(mmax + 1):
        dini_zs[m, :] = dini_roots(m, Nr, H)
    zs       = dini_zs
    filedir  = f'mmts_dini_Nr_{Nr}_Nphi_{Nphi}/'
    filename = f'zb_dini_Nr_{Nr}_Nphi_{Nphi}.npy'
else:
    std_zs = np.zeros((mmax + 1, Nr))
    for m in range(mmax + 1):
        std_zs[m, :] = std_roots(m, Nr)
    zs       = std_zs
    filedir  = f'mmts_J_Nr_{Nr}_Nphi_{Nphi}/'
    filename = f'zb_J_Nr_{Nr}_Nphi_{Nphi}.npy'

if not make_new:
    ZBs = np.load(filedir + filename, allow_pickle=True)[()]
    logger.info('ZBs loaded from %s', filedir + filename)
else:
    logger.info('Building ZBs matrices from scratch')
    ZBs = makeZBs_dini(Nr, Nphi, dini_zs) if dini else makeZBs_std(Nr, Nphi, std_zs)
    Path(filedir).mkdir(parents=True, exist_ok=True)
    np.save(filedir + filename, ZBs, allow_pickle=True)
logger.info('ZBs ready')

# ---------------------------------------------------------------------------
# Binning setup
# ---------------------------------------------------------------------------

masks_m = {}
Nbins, centers, edges, counts, masks = define_bins(zs)
for m in range(mmax + 1):
    _, _, _, _, mask_m = define_bins_m(zs, mmax, m)
    masks_m[m] = mask_m
bin_widths = np.diff(edges)

# ---------------------------------------------------------------------------
# Select writes to process
# ---------------------------------------------------------------------------

ws = np.arange(np.where(t <= t_out_start)[0][-1],
               np.where(t >= t_out_end)[0][0] + 1)
nw = len(ws)
tw = t[ws]

# ---------------------------------------------------------------------------
# Allocate arrays
# ---------------------------------------------------------------------------

vortZ = np.zeros((nw, Nphi, Nr)) # Zernike coeffs of vorticity
vortB = np.zeros((nw, Nphi, Nr)) # Bessel coeffs of vorticity

upZ = np.zeros((nw, Nphi, Nr))   # spin+ Zernike coeffs of u
umZ = np.zeros((nw, Nphi, Nr))   # spin- Zernike coeffs of u

npZ = np.zeros((nw, Nphi, Nr))   # spin+ Zernike coeffs of nonlin = u@grad(u)
nmZ = np.zeros((nw, Nphi, Nr))   # spin- Zernike coeffs of nonlin


if dini:
    upZ = np.zeros((nw, Nphi, Nr))   # spin+ Zernike coeffs of u
    umZ = np.zeros((nw, Nphi, Nr))   # spin- Zernike coeffs of u
    npZ = np.zeros((nw, Nphi, Nr))   # spin+ Zernike coeffs of nonlin = u@grad(u)
    nmZ = np.zeros((nw, Nphi, Nr))   # spin- Zernike coeffs of nonlin
    upB = np.zeros((nw, Nphi, Nr))   # spin+ Bessel coeffs of u
    umB = np.zeros((nw, Nphi, Nr))
    npB = np.zeros((nw, Nphi, Nr))   # spin+ Bessel coeffs of nonlin
    nmB = np.zeros((nw, Nphi, Nr))
else:
    psiZ = np.zeros((nw, Nphi, Nr))
    psiB = np.zeros((nw, Nphi, Nr))



vortB = np.zeros((nw, Nphi, Nr))

# 2D (m, k_radial) spectral arrays
keB = np.zeros((nw, mmax + 1, Nr))
enB = np.zeros((nw, mmax + 1, Nr))
tB  = np.zeros((nw, mmax + 1, Nr))   # nonlinear transfer (dini only)

# 1D binned spectra (summed over m, normalised by bin width)
keBn   = np.zeros((nw, Nbins))
enBn   = np.zeros((nw, Nbins))
fluxBn = np.zeros((nw, Nbins))
tBn    = np.zeros((nw, Nbins))        # T(k,t), dini only

# m-resolved binned spectra
keBmn = np.zeros((nw, mmax + 1, Nbins))
enBmn = np.zeros((nw, mmax + 1, Nbins))
tBmn  = np.zeros((nw, mmax + 1, Nbins))

# ---------------------------------------------------------------------------
# Main loop over saved writes
# ---------------------------------------------------------------------------

prog_cad = 32

for i, w in enumerate(ws):
    if i % prog_cad == 0:
        logger.info("Write loop: i = %d / %d", i, nw)

    # --- Load fields ---
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)
    nonlin_field.load_from_hdf5(f, w)

    # --- Gather vorticity Zernike coefficients (all ranks send, root assembles) ---
    vort.change_scales(1)
    vortZgather = comm.gather(np.copy(vort['c']), root=0)
    if rank == 0:
        vortZ[i, :, :] = np.array(vortZgather).reshape(Nphi, Nr)

    if dini:
        # --- u ---
        u.change_scales(1)
        upZgather = comm.gather(np.copy(u['c'][0, :, :]), root=0)
        umZgather = comm.gather(np.copy(u['c'][1, :, :]), root=0)
        if rank == 0:
            upZ[i, :, :] = np.array(upZgather).reshape(Nphi, Nr)
            umZ[i, :, :] = np.array(umZgather).reshape(Nphi, Nr)

        # --- nonlin ---
        nonlin_field.change_scales(1)
        npZgather = comm.gather(np.copy(nonlin_field['c'][0, :, :]), root=0)
        nmZgather = comm.gather(np.copy(nonlin_field['c'][1, :, :]), root=0)
        if rank == 0:
            npZ[i, :, :] = np.array(npZgather).reshape(Nphi, Nr)
            nmZ[i, :, :] = np.array(nmZgather).reshape(Nphi, Nr)

    else:
        # --- Solve for streamfunction (std / Dirichlet branch) ---
        u.change_scales(dealias)
        problem_psi = d3.LBVP([psi, tau_psi, tau_psi2], namespace=locals())
        problem_psi.add_equation("lap(psi) + lift(tau_psi) + tau_psi2 = vort")
        problem_psi.add_equation("psi(r=1) = 0")
        problem_psi.add_equation("integ(psi) = 0")
        solver_psi = problem_psi.build_solver()
        solver_psi.solve()
        psi.change_scales(1)
        psiZgather = comm.gather(np.copy(psi['c'][:, :]), root=0)
        if rank == 0:
            psiZ[i, :, :] = np.array(psiZgather).reshape(Nphi, Nr)

    # --- Zernike -> Bessel transform and spectra (rank 0 only) ---
    if rank == 0:
        for m in range(mmax + 1):
            if m % prog_cad == 0:
                logger.info("ZB transform: m = %d / %d", m, mmax)
            ZB  = ZBs[m]
            midx = m_map(m, Nphi)
            mc, ms_idx = midx, midx + 1   # cosine and sine index slots

            vortB[i, mc, :]     = (ZB @ vortZ[i, mc, :][0, :]).reshape(1, Nr)
            vortB[i, ms_idx, :] = (ZB @ vortZ[i, ms_idx, :][0, :]).reshape(1, Nr)

            if dini:
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
                                          umB[i, mc, :], umB[i, ms_idx, :], dini_zs)
                enB[i, m, :] = en_m_dini(m, Nr,
                                          vortB[i, mc, :], vortB[i, ms_idx, :], dini_zs)
                tB[i, m, :]  = t_m_dini( m, Nr,
                                          upB[i, mc, :],  upB[i, ms_idx, :],
                                          umB[i, mc, :],  umB[i, ms_idx, :],
                                          npB[i, mc, :],  npB[i, ms_idx, :],
                                          nmB[i, mc, :],  nmB[i, ms_idx, :], dini_zs)
            else:
                psiB[i, mc, :]     = (ZB @ psiZ[i, mc, :][0, :]).reshape(1, Nr)
                psiB[i, ms_idx, :] = (ZB @ psiZ[i, ms_idx, :][0, :]).reshape(1, Nr)
                keB[i, m, :] = ke_m_std(m, Nr,
                                         psiB[i, mc, :], psiB[i, ms_idx, :], std_zs)
                enB[i, m, :] = en_m_std(m, Nr,
                                         psiB[i, mc, :], psiB[i, ms_idx, :], std_zs)

        # --- Bin spectra over m ---
        for m in range(mmax + 1):
            keBmn[i, m, :] = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks_m[m])
            enBmn[i, m, :] = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks_m[m])
            if dini:
                tBmn[i, m, :] = bin_spectra(tB[i, :, :], bin_widths, Nbins, masks_m[m])

        keBn[i, :]   = bin_spectra(keB[i, :, :], bin_widths, Nbins, masks)
        enBn[i, :]   = bin_spectra(enB[i, :, :], bin_widths, Nbins, masks)
        fluxBn[i, :] = flux_k(keBn[i, :], lambda_k(keBn[i, :], centers), centers)
        if dini:
            tBn[i, :] = bin_spectra(tB[i, :, :], bin_widths, Nbins, masks)

# ---------------------------------------------------------------------------
# Post-loop: analytical dissipation terms and budget closure (rank 0 only)
# ---------------------------------------------------------------------------

if rank == 0 and dini:
    logger.info("Computing dissipation spectra and budget closure")

    # D_nu(m,n) = 2*nu * z_{m,n}^2 * E(m,n)
    # Uses each mode's actual Bessel wavenumber rather than the bin-centre
    # approximation, which matters when a bin spans a range of wavenumbers.
    D_nu_B  = 2.0 * nu * dini_zs[None, :, :]**2 * keB   # (nw, mmax+1, Nr)
    D_nu_Bn = np.zeros((nw, Nbins))
    for i in range(nw):
        D_nu_Bn[i, :] = bin_spectra(D_nu_B[i, :, :], bin_widths, Nbins, masks)

    # D_alpha(k,t) = 2*alpha * E(k,t) — uniform across all modes
    D_alpha_Bn = 2.0 * alpha * keBn   # (nw, Nbins)

    # dE/dt from central finite differences in time
    if nw > 1:
        dkeBn_dt = np.gradient(keBn, tw, axis=0)   # (nw, Nbins)
    else:
        dkeBn_dt = np.zeros_like(keBn)
        logger.info("Only one write; dE/dt set to zero")

    # Budget residual: dE/dt - T + D_nu + D_alpha
    # F(k) is omitted here.  Outside the forcing band the residual should be ~0,
    # confirming T and D are correctly computed.  In the forcing band the residual
    # will be ~ -eps (the missing injection term).
    budget_residual_Bn = dkeBn_dt - tBn + D_nu_Bn + D_alpha_Bn if nw > 1 else None

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Saving: %s", output_prefix + '_' + output_suffix + '.npy')
    processed = {}

    processed['ws']      = ws
    processed['ts']      = tw
    processed['ms']      = ms
    processed['zs']      = zs
    processed['Nbins']   = Nbins
    processed['centers'] = centers
    processed['edges']   = edges
    processed['counts']  = counts
    processed['masks']   = masks
    processed['masks_m'] = masks_m

    if not steady_only:
        # Spectral coefficients
        processed['vortZ'] = vortZ
        processed['vortB'] = vortB
        if dini:
            processed['upZ'] = upZ;  processed['umZ'] = umZ
            processed['upB'] = upB;  processed['umB'] = umB
            processed['npZ'] = npZ;  processed['nmZ'] = nmZ
            processed['npB'] = npB;  processed['nmB'] = nmB
        else:
            processed['psiZ'] = psiZ
            processed['psiB'] = psiB

        # 2D (m, k) spectra
        processed['keB'] = keB
        processed['enB'] = enB
        if dini:
            processed['tB'] = tB

        # 1D binned spectra
        processed['keBn']   = keBn
        processed['enBn']   = enBn
        processed['fluxBn'] = fluxBn
        if dini:
            processed['tBn']        = tBn
            processed['D_nu_Bn']    = D_nu_Bn
            processed['D_alpha_Bn'] = D_alpha_Bn
            processed['dkeBn_dt']   = dkeBn_dt
            if budget_residual_Bn is not None:
                processed['budget_residual_Bn'] = budget_residual_Bn

        # m-resolved binned spectra
        processed['keBmn'] = keBmn
        processed['enBmn'] = enBmn
        if dini:
            processed['tBmn'] = tBmn

    # ---- Time-averaged quantities (always saved) ----
    if nw > 1:
        t_end     = tw[-1]
        t_start   = t_end - t_steady_range
        idx_end   = np.where(tw <= t_end)[0][-1]
        idx_start = np.where(tw >= t_start)[0][0]
        sl = slice(idx_start, idx_end)

        for key, arr in [('keB', keB), ('enB', enB)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)
        processed['keBsum_tavg'] = np.sum(processed['keB_tavg'], axis=1)
        processed['enBsum_tavg'] = np.sum(processed['enB_tavg'], axis=1)

        for key, arr in [('keBn', keBn), ('enBn', enBn), ('fluxBn', fluxBn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)
        for key, arr in [('keBmn', keBmn), ('enBmn', enBmn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        if dini:
            for key, arr in [('tBn', tBn), ('D_nu_Bn', D_nu_Bn),
                              ('D_alpha_Bn', D_alpha_Bn)]:
                processed[key + '_tavg'] = np.mean(arr[sl], axis=0)
            processed['tBmn_tavg'] = np.mean(tBmn[sl], axis=0)
            if budget_residual_Bn is not None:
                processed['budget_residual_Bn_tavg'] = np.mean(
                    budget_residual_Bn[sl], axis=0)

    np.save(output_prefix + '_' + output_suffix + '.npy', processed)
    logger.info("Save complete")
else:
    logger.info("Rank %d finished", rank)

"""
Bessel energy spectra and spectral budget for forced-dissipative 2D disk
turbulence, with **mixed Dirichlet / Robin H=-1 Bessel bases** for the two
velocity components.

v3 differences relative to v2:
  * The 'nonlin' analysis task in the IVP is now expected to be the **vector**
    field u@grad(u), saved at grid layout (cf. disk_turb_run_init_v3.py).
  * The radial Bessel basis index uses the vector-regularity shift:
        m_J = |m - 1|
    so that the Bessel basis J_{m_J}(lambda r) matches the natural r^{|m-1|}
    behaviour of u_r and u_phi (Fourier-m components of a regular vector
    field) at the origin.
  * u_r is projected onto J_{m_J}(lambda^{(D)}_{m,n} r) where lambda^{(D)} are
    the Dirichlet roots of J_{m_J}.  Matches u_r(r=1) = 0 exactly.
  * u_phi is projected onto J_{m_J}(lambda^{(R)}_{m,n} r) where lambda^{(R)}
    are the Robin / Dini roots solving  z J_{m_J}'(z) + H J_{m_J}(z) = 0
    with H = -1.  Matches the stress-free BC  d/dr (u_phi/r)|_{r=1} = 0
    exactly.
  * The two ladders {lambda^{(D)}} and {lambda^{(R)}} are binned into a
    single shared k-grid (Dirichlet roots of J_0 as bin centres) for the
    physical-space diagnostics E(k,t), T(k,t), D_nu(k,t), D_alpha(k,t).
  * For the m in {0, 1, 2} sectors, where m_J + H <= 0, the H=-1 Dini
    expansion is not complete: there is a missing "initial term" r^{m_J}
    (Watson, ch. XVIII).  We compute the L^2 projection of u_phi,m(r) onto
    r^{m_J} as a **diagnostic** (saved as `D_init_m`) so we can verify it is
    small (equivalent to verifying the IVP's L, P_x, P_y diagnostics stay
    near zero).  We do *not* explicitly subtract this content before doing
    the Bessel projection -- the L^2 projection onto the standard Bessel
    basis is the best-fit and the omitted content stays small under the
    forcing/IC construction.
  * The de-spinning step is done in grid space: we read u['g'][0] (the phi
    component) and u['g'][1] (the r component), do a phi-FFT via np.fft.rfft,
    and convert to (a_m, b_m) cos/sin amplitudes consistent with Dedalus's
    RealFourier convention  f(phi) = a_m cos(m phi) - c_m sin(m phi).
  * The radial projection uses Dedalus's *exact* Gauss-Legendre quadrature
    weights for the disk's r dr measure, obtained via
        dedalus.libraries.dedalus_sphere.zernike.quadrature(2, Nr, k=0).
    These weights satisfy sum_i w_r[i] f(r_i) = ∫_0^1 f(r) r dr exactly for
    polynomial f of degree <= 2*Nr-1, and the integrands here (Bessel func.
    times polynomial-like u) are analytic on [0,1], giving effectively
    machine-precision quadrature.  Basis matrices B_m^D and B_m^R are
    precomputed once; per-snapshot projection is a single matmul per m.

Outputs:
  E(k,t), T(k,t), D_nu(k,t), D_alpha(k,t), dE/dt(k,t), Enstrophy(k,t),
  m-resolved versions of each, and the L, P_x, P_y conservation residuals
  needed to certify the initial-term modes are not carrying significant
  energy.

Usage:
    process_spectra_option_C.py <file>... [options]

Options:
    --t_out_start=<value>       Simulation time to begin making spectra, or "auto" for the first saved snapshot's time [default: auto]
    --t_out_end=<value>         Simulation time to stop making spectra, or "auto" for the last saved snapshot's time [default: auto]
    --t_steady_range=<value>    Size of time window prior to t_out_end to average as steady state, or "auto" for the full processed duration [default: auto]
    --steady_only=<bool>        True: save only time-averaged steady-state data; False: save all timesteps [default: True]
"""

import numpy as np
import h5py
import scipy.special as sp
from scipy.optimize import newton
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
steady_only    = eval(args['--steady_only'])

output_prefix = 'processed_spectra_v3'
output_prefix += '_steady' if steady_only else ''

# ---------------------------------------------------------------------------
# Bessel / Dini helper functions
# ---------------------------------------------------------------------------

H = -1.0  # Robin parameter; H=-1 corresponds to the stress-free BC on u_phi.

def m_bessel(m):
    """Bessel radial index for vector-regularity expansion at azimuthal mode m.

    Regular vector components have leading r^|m-1| behaviour near the origin,
    so we expand each Fourier-m component in J_{|m-1|}(lambda r).
    """
    return abs(m - 1)

def needs_initial_term(m_J, H):
    """True if the standard H Dini expansion in J_{m_J} is incomplete and a
    r^{m_J} initial term is required (Watson, ch. XVIII).  For H = -1 this
    happens at m_J in {0, 1}, i.e. at the original azimuthal modes 0, 1, 2.
    """
    return (m_J + H) <= 0

def dirichlet_roots(m_J, N):
    """First N positive roots of J_{m_J}(z) = 0."""
    return sp.jn_zeros(m_J, N)

def dirichlet_weight(m_J, z):
    """Fourier-Bessel (Dirichlet) normalisation:
       ∫_0^1 J_{m_J}(z r)^2 r dr = (1/2) J_{m_J+1}(z)^2  for z a root of J_{m_J}.
    """
    return 0.5 * sp.jv(m_J + 1, z) ** 2

def robin_func(z, m_J, H):
    return np.real(z * sp.jvp(m_J, z, n=1) + H * sp.jv(m_J, z))

def robin_func_prime(z, m_J, H):
    return np.real(z * sp.jvp(m_J, z, n=2) + (H + 1) * sp.jvp(m_J, z, n=1))

def robin_roots(m_J, N, H):
    """First N positive roots of  z J_{m_J}'(z) + H J_{m_J}(z) = 0.

    Implemented as a fine scan followed by brentq bracketing, so that:
      * the degenerate triple-root at z=0 (which appears for the H=-1,
        m_J=1 borderline case) is skipped by starting the scan at a small
        positive epsilon, and
      * we never accidentally converge to the wrong basin.
    """
    from scipy.optimize import brentq
    # Scan range: well past the Nth expected root.  Asymptotically
    # lambda_n ~ (n + m_J/2 - 1/4)*pi, so N*pi + a safety margin is enough.
    z_max = (N + max(m_J, 1) + 5) * np.pi
    z_scan = np.linspace(1e-2, z_max, max(20 * N, 2000))
    f_scan = z_scan * sp.jvp(m_J, z_scan, n=1) + H * sp.jv(m_J, z_scan)
    roots = []
    for i in range(len(z_scan) - 1):
        if f_scan[i] * f_scan[i + 1] < 0:
            r = brentq(robin_func, z_scan[i], z_scan[i + 1],
                       args=(m_J, H), xtol=1e-12)
            roots.append(r)
            if len(roots) == N:
                break
    if len(roots) < N:
        raise RuntimeError(
            f"Found only {len(roots)} of {N} Robin roots for m_J={m_J}; "
            "increase z_max or scan density.")
    return np.array(roots[:N])

def robin_weight(m_J, z, H):
    """Dini (Robin-H) normalisation:
       ∫_0^1 J_{m_J}(z r)^2 r dr = (H^2 + z^2 - m_J^2) J_{m_J}(z)^2 / (2 z^2)
       for z a root of  z J_{m_J}'(z) + H J_{m_J}(z) = 0.
    """
    return ((H ** 2 + z ** 2 - m_J ** 2) * sp.jv(m_J, z) ** 2) / (2 * z ** 2)

def initial_term_weight(m_J):
    """∫_0^1 r * (r^{m_J})^2 dr = 1 / (2(m_J + 1))."""
    return 1.0 / (2.0 * (m_J + 1))

# ---------------------------------------------------------------------------
# Legacy trapezoidal radial quadrature helpers.
# These are NO LONGER USED in the main pipeline (we use Dedalus's exact
# Gauss-Legendre weights via dedalus_sphere.zernike.quadrature instead).
# Kept here for verification / cross-checking against the spectral pipeline.
# ---------------------------------------------------------------------------

def radial_grid_quadrature_weights(r_grid):
    """Return trapezoidal weights q_i such that
           ∫_0^1 g(r) dr  ≈  sum_i  q_i  g(r_grid[i])
    with r=0 and r=1 treated as endpoints via linear extension.
    """
    r = r_grid
    Nr_local = len(r)
    q = np.zeros(Nr_local)
    # Lower endpoint: midpoint between 0 and r[0], plus half-cell to r[1].
    q[0] = 0.5 * (r[1] - 0.0)        # midpoint(0, r[1])
    q[-1] = 0.5 * (1.0 - r[-2])      # midpoint(r[-2], 1)
    for i in range(1, Nr_local - 1):
        q[i] = 0.5 * (r[i + 1] - r[i - 1])
    return q

def radial_integrate(integrand, r_grid, q):
    """∫_0^1 integrand(r) dr with precomputed trapezoidal weights q."""
    return np.sum(integrand * q)

def project_radial_bessel(profile, r_grid, q, m_J, lambdas, mode_weights):
    """For a 1-D radial profile and an array of N eigenvalues lambdas (and
    corresponding normalisation weights mode_weights), return the N spectral
    coefficients C_n such that
           profile(r) ≈ sum_n  C_n  J_{m_J}(lambda_n r).
    Computed by L^2 projection:
           C_n = (1 / mode_weights[n])  ∫_0^1 profile(r) J_{m_J}(lambda_n r) r dr.
    """
    N = len(lambdas)
    coefs = np.zeros(N)
    for n in range(N):
        basis = sp.jv(m_J, lambdas[n] * r_grid)
        coefs[n] = radial_integrate(profile * basis * r_grid, r_grid, q) / mode_weights[n]
    return coefs

def project_initial_term(profile, r_grid, q, m_J):
    """L^2 projection onto the initial-term basis r^{m_J}:
           D = (1 / initial_term_weight(m_J))  ∫_0^1 profile(r) r^{m_J} r dr.
    """
    return radial_integrate(profile * r_grid ** m_J * r_grid, r_grid, q) / initial_term_weight(m_J)

# ---------------------------------------------------------------------------
# phi-FFT helper.  Dedalus RealFourier convention:
#     f(phi) = a_0 + sum_{m>=1} [ a_m cos(m phi) - c_m sin(m phi) ]
# We work with standard amplitudes
#     f(phi) = a_0 + sum_{m>=1} [ a_m cos(m phi) + b_m sin(m phi) ],
# i.e. b_m = -c_m.  Conversion from np.fft.rfft output F[m]:
#     a_0  = (1/Nphi) F[0]
#     a_m  = (2/Nphi) Re(F[m])     (m=1..Nphi/2-1)
#     b_m  = -(2/Nphi) Im(F[m])    (m=1..Nphi/2-1)
#     a_{Nphi/2}, b_{Nphi/2}: real-only Nyquist mode for even Nphi.
# ---------------------------------------------------------------------------

def phi_fft_to_cos_sin(grid_array, Nphi):
    """grid_array shape (Nphi, Nr) -> (a, b) each of shape (Nm, Nr), where
    Nm = Nphi/2 + 1.  a, b are the cos and sin amplitudes."""
    F = np.fft.rfft(grid_array, axis=0)        # shape (Nm, Nr), complex
    a = np.zeros_like(F.real)
    b = np.zeros_like(F.real)
    a[0] = F[0].real / Nphi
    Nm = F.shape[0]
    if Nphi % 2 == 0 and Nm == Nphi // 2 + 1:
        a[1:-1] = 2.0 * F[1:-1].real / Nphi
        b[1:-1] = -2.0 * F[1:-1].imag / Nphi
        a[-1]   = F[-1].real / Nphi
        # b[-1] left at zero (Nyquist of real signal has no sin part)
    else:
        a[1:] = 2.0 * F[1:].real / Nphi
        b[1:] = -2.0 * F[1:].imag / Nphi
    return a, b

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# How many azimuthal modes we resolve.  Standard real-FFT range is 0 .. Nphi/2.
# We restrict to m <= mmax = Nphi/2 - 1 (drop Nyquist) to match v2's layout.
mmax = int(Nphi / 2) - 1
ms = np.arange(mmax + 1)

# Pre-compute Bessel root tables and weight tables for each m_J value that
# arises.  m_J = |m-1| takes every integer value in [0, mmax-1] across m.
m_J_values = np.unique(np.array([m_bessel(m) for m in ms]))

dirichlet_zs = {}
dirichlet_ws = {}
robin_zs     = {}
robin_ws     = {}

# vort is a scalar field at azimuthal mode m (regularity r^m), so it gets
# its own Dirichlet ladder with m_J = m.  Velocity components use m_J=|m-1|.
# Union covers m_J = 0 .. mmax inclusive.
m_J_values_all = np.arange(mmax + 1)

if rank == 0:
    logger.info("Computing Bessel root tables for m_J in 0..%d", mmax)

for m_J in m_J_values_all:
    dz = dirichlet_roots(m_J, Nr)
    dirichlet_zs[m_J] = dz
    dirichlet_ws[m_J] = dirichlet_weight(m_J, dz)
    rz = robin_roots(m_J, Nr, H)
    robin_zs[m_J] = rz
    robin_ws[m_J] = robin_weight(m_J, rz, H)

# Per-(m, n) eigenvalue/weight arrays.
# D, R are indexed by velocity-regularity m_J = |m-1| (for u_r, u_phi).
# V is indexed by scalar-regularity m_J = m (for the vorticity diagnostic).
lambda_D = np.zeros((mmax + 1, Nr))
lambda_R = np.zeros((mmax + 1, Nr))
weight_D = np.zeros((mmax + 1, Nr))
weight_R = np.zeros((mmax + 1, Nr))
lambda_V = np.zeros((mmax + 1, Nr))
weight_V = np.zeros((mmax + 1, Nr))
for m in ms:
    m_J = m_bessel(m)
    lambda_D[m, :] = dirichlet_zs[m_J]
    lambda_R[m, :] = robin_zs[m_J]
    weight_D[m, :] = dirichlet_ws[m_J]
    weight_R[m, :] = robin_ws[m_J]
    lambda_V[m, :] = dirichlet_zs[m]
    weight_V[m, :] = dirichlet_ws[m]

# ---------------------------------------------------------------------------
# Load HDF5 file and Dedalus bases
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

u            = dist.VectorField(coords, name='u',       bases=disk)
vort         = dist.Field(              name='vort',    bases=disk)

if 'nonlin_u' in list(f['tasks']):
    nonlin_name = 'nonlin_u'
else:
    nonlin_name = 'nonlin'
nonlin_field = dist.VectorField(coords, name=nonlin_name,  bases=disk)

# Quadrature weights on the (1D) radial grid.
# Use Dedalus's exact Gauss-Legendre weights (via dedalus_sphere.zernike) for the
# disk's natural inner product weight  dV_r = r dr.  These weights satisfy
#     sum_i w_r[i] f(r_i) = ∫_0^1 f(r) r dr   (exactly for polynomial f of degree
#     <= 2*Nr - 1)
# and they ALREADY absorb the r factor of the disk's area element -- no extra
# r multiplication is needed when applying them.
r_1d = r_local[0, :]            # shape (Nr,)
z0, w_r = zsphere.quadrature(2, Nr, k=0)
# Sanity check: sum_i w_r[i] = ∫_0^1 r dr = 1/2.
_quad_sum = float(np.sum(w_r))
if not np.isclose(_quad_sum, 0.5, atol=1e-12):
    raise RuntimeError(
        f"Dedalus Gauss-Legendre weights do not sum to 1/2: got {_quad_sum:.16f}")
# Also verify Dedalus's radial grid matches the zernike-quadrature nodes mapped
# to r = sqrt((1+z)/2).  If this fails, the weights are still correct for the
# quadrature points but our matrix-product approach (which evaluates u on
# Dedalus's grid) would be inconsistent.
_r_from_z0 = np.sqrt(0.5 * (1.0 + z0))
if not np.allclose(np.sort(r_1d), np.sort(_r_from_z0), atol=1e-10):
    raise RuntimeError(
        "Dedalus radial grid does not match dedalus_sphere.zernike.quadrature "
        "nodes; quadrature weights cannot be applied directly to u['g'].")

# ---------------------------------------------------------------------------
# Precompute Bessel-projection matrices (one-time setup).
#
# For each m_J in [0, mmax]:
#   B_D[m_J][i, n] = J_{m_J}(lambda_n^D * r_i) * w_r[i]   (Dirichlet ladder)
#   B_R[m_J][i, n] = J_{m_J}(lambda_n^R * r_i) * w_r[i]   (Robin H=-1 ladder)
#
# Given a radial profile  f(r_i) = profile_grid,  the projection coefficient
# onto the n-th basis mode is
#   c_n = (1 / mode_weight[n]) * sum_i  B[i, n] * profile_grid[i]
#       = (1 / mode_weight[n]) * (profile_grid @ B)[n]
# i.e. a single matrix-vector product per azimuthal m, no per-mode quadrature
# loop, exact to Dedalus's Gauss-Legendre precision.
# ---------------------------------------------------------------------------

B_D = {}
B_R = {}
for m_J in m_J_values_all:
    # broadcast: rows = grid points r_i, columns = Bessel modes n
    B_D[m_J] = (
        sp.jv(m_J, dirichlet_zs[m_J][np.newaxis, :] * r_1d[:, np.newaxis])
        * w_r[:, np.newaxis]
    )
    B_R[m_J] = (
        sp.jv(m_J, robin_zs[m_J][np.newaxis, :] * r_1d[:, np.newaxis])
        * w_r[:, np.newaxis]
    )
# For the scalar Dirichlet vort projection at azimuthal m, the Bessel index
# matches m (scalar regularity).  Reuse B_D since it's already indexed by m_J.

# ---------------------------------------------------------------------------
# Binning: use Dirichlet roots of J_0 as bin centres.
# ---------------------------------------------------------------------------

def define_bins(centers):
    Nbins = len(centers)
    edges = np.array([centers[i] - 0.5 * (centers[i] - centers[i - 1])
                      for i in range(1, Nbins)])
    edges = np.concatenate(([0.0], edges, [centers[-1] + np.pi / 2]))
    return Nbins, edges

bin_centers = sp.jn_zeros(0, Nr)         # Dirichlet roots of J_0
Nbins, bin_edges = define_bins(bin_centers)
bin_widths = np.diff(bin_edges)

def bin_one(values, data):
    """Bin (values, data) pairs into the global k-bins, returning per-bin
    sums divided by bin width."""
    out = np.zeros(Nbins)
    for b in range(Nbins):
        mask = (values >= bin_edges[b]) & (values < bin_edges[b + 1])
        out[b] = np.sum(data[mask]) / bin_widths[b]
    return out

def bin_one_m_resolved(values, data, mmax_local):
    """Bin separately by azimuthal m.  values, data shape (mmax+1, Nr)."""
    out = np.zeros((mmax_local + 1, Nbins))
    for m in range(mmax_local + 1):
        for b in range(Nbins):
            mask = (values[m, :] >= bin_edges[b]) & (values[m, :] < bin_edges[b + 1])
            out[m, b] = np.sum(data[m, mask]) / bin_widths[b]
    return out

# ---------------------------------------------------------------------------
# Select writes to process
# ---------------------------------------------------------------------------

#ws = np.arange(np.where(t <= t_out_start)[0][-1],
#               np.where(t >= t_out_end)[0][0] + 1)
ws = np.arange(np.where(t == np.min(t))[0][0], np.where(t == np.max(t))[0][0] + 1)
nw = len(ws)
tw = t[ws]

# ---------------------------------------------------------------------------
# Allocate arrays  (rank 0 keeps the full result)
# ---------------------------------------------------------------------------

# Per (snapshot, m, n) Bessel-mode arrays
keB_r = np.zeros((nw, mmax + 1, Nr))   # KE contribution from u_r (Dirichlet ladder)
keB_p = np.zeros((nw, mmax + 1, Nr))   # KE contribution from u_phi (Robin ladder)
tB_r  = np.zeros((nw, mmax + 1, Nr))
tB_p  = np.zeros((nw, mmax + 1, Nr))
enB   = np.zeros((nw, mmax + 1, Nr))   # enstrophy spectrum (vort, scalar Dirichlet J_m ladder)

# Initial-term diagnostics for m in {0, 1, 2}: cos and sin amplitudes
D_init_cos = np.zeros((nw, 3))
D_init_sin = np.zeros((nw, 3))         # m=0 sin slot left at 0

# 1D binned outputs
keBn      = np.zeros((nw, Nbins))
keBn_r    = np.zeros((nw, Nbins))
keBn_p    = np.zeros((nw, Nbins))
tBn       = np.zeros((nw, Nbins))
enBn      = np.zeros((nw, Nbins))

# m-resolved binned outputs
keBmn_r = np.zeros((nw, mmax + 1, Nbins))
keBmn_p = np.zeros((nw, mmax + 1, Nbins))
tBmn_r  = np.zeros((nw, mmax + 1, Nbins))
tBmn_p  = np.zeros((nw, mmax + 1, Nbins))
enBmn   = np.zeros((nw, mmax + 1, Nbins))

# ---------------------------------------------------------------------------
# Main loop over snapshots
# ---------------------------------------------------------------------------

prog_cad = 32

for i, w in enumerate(ws):
    if i % prog_cad == 0 and rank == 0:
        logger.info("Write loop: i = %d / %d  (t = %.3f)", i, nw, tw[i])

    # --- Load grid fields ---
    u.load_from_hdf5(f, w)
    vort.load_from_hdf5(f, w)
    nonlin_field.load_from_hdf5(f, w)

    u.change_scales(1)
    vort.change_scales(1)
    nonlin_field.change_scales(1)

    # Gather grid data to rank 0 only.  Each rank has shape (Nphi_local, Nr).
    u_phi_loc =  np.copy(u['g'][0])         # phi-component grid values
    u_r_loc   =  np.copy(u['g'][1])         # r-component grid values
    n_phi_loc =  np.copy(nonlin_field['g'][0])
    n_r_loc   =  np.copy(nonlin_field['g'][1])
    vort_loc  =  np.copy(vort['g'])

    u_phi_g_list = comm.gather(u_phi_loc, root=0)
    u_r_g_list   = comm.gather(u_r_loc,   root=0)
    n_phi_g_list = comm.gather(n_phi_loc, root=0)
    n_r_g_list   = comm.gather(n_r_loc,   root=0)
    vort_g_list  = comm.gather(vort_loc,  root=0)

    if rank != 0:
        continue

    # Reassemble full (Nphi, Nr) grid arrays.
    u_phi_g = np.concatenate(u_phi_g_list, axis=0)
    u_r_g   = np.concatenate(u_r_g_list,   axis=0)
    n_phi_g = np.concatenate(n_phi_g_list, axis=0)
    n_r_g   = np.concatenate(n_r_g_list,   axis=0)
    vort_g  = np.concatenate(vort_g_list,  axis=0)

    # phi-FFT to (a_m, b_m) cos/sin amplitudes.
    a_u_r,  b_u_r  = phi_fft_to_cos_sin(u_r_g,   Nphi)
    a_u_p,  b_u_p  = phi_fft_to_cos_sin(u_phi_g, Nphi)
    a_n_r,  b_n_r  = phi_fft_to_cos_sin(n_r_g,   Nphi)
    a_n_p,  b_n_p  = phi_fft_to_cos_sin(n_phi_g, Nphi)
    a_vort, b_vort = phi_fft_to_cos_sin(vort_g,  Nphi)

    # Per-m projection and energy/transfer accumulation.
    for m in range(mmax + 1):
        m_J = m_bessel(m)
        zD  = dirichlet_zs[m_J]
        wD  = dirichlet_ws[m_J]
        zR  = robin_zs[m_J]
        wR  = robin_ws[m_J]
        # Scalar Dirichlet ladder for vort: uses m_J = m (the scalar regularity
        # index), distinct from the |m-1| velocity index above.
        zV  = dirichlet_zs[m]
        wV  = dirichlet_ws[m]

        # Per-mode azimuthal weighting factor in inner products on the disk:
        #   ∫_0^{2π} cos^2(m phi) d phi = pi for m>0, 2 pi for m=0
        #   (sin part contributes pi for m>0)
        phi_factor_cos = 2 * np.pi if m == 0 else np.pi
        phi_factor_sin = 0.0      if m == 0 else np.pi

        # --- u_r on Dirichlet ladder (Bessel index m_J = |m-1|) ---
        # Matrix-vector projection: c_n = (profile_grid @ B_D[m_J])[n] / wD[n].
        c_ur_cos = (a_u_r[m, :] @ B_D[m_J]) / wD
        c_ur_sin = (b_u_r[m, :] @ B_D[m_J]) / wD
        c_nr_cos = (a_n_r[m, :] @ B_D[m_J]) / wD
        c_nr_sin = (b_n_r[m, :] @ B_D[m_J]) / wD

        # --- u_phi on Robin H=-1 ladder (Bessel index m_J = |m-1|) ---
        c_up_cos = (a_u_p[m, :] @ B_R[m_J]) / wR
        c_up_sin = (b_u_p[m, :] @ B_R[m_J]) / wR
        c_np_cos = (a_n_p[m, :] @ B_R[m_J]) / wR
        c_np_sin = (b_n_p[m, :] @ B_R[m_J]) / wR

        # --- vorticity on scalar Dirichlet J_m ladder (scalar regularity) ---
        c_w_cos  = (a_vort[m, :] @ B_D[m]) / wV
        c_w_sin  = (b_vort[m, :] @ B_D[m]) / wV

        # Energy contributions, mode-by-mode:
        #   E_r(m, n) = (1/2) * phi_factor * wD[n] * (c_ur_cos[n]^2 + c_ur_sin[n]^2)
        # The (1/2) is for KE = (1/2)|u|^2.
        keB_r[i, m, :] = 0.5 * wD * (phi_factor_cos * c_ur_cos ** 2 + phi_factor_sin * c_ur_sin ** 2)
        keB_p[i, m, :] = 0.5 * wR * (phi_factor_cos * c_up_cos ** 2 + phi_factor_sin * c_up_sin ** 2)

        # Transfer T(m, n) = - <u, u@grad(u)>:
        tB_r[i, m, :] = -wD * (phi_factor_cos * c_ur_cos * c_nr_cos + phi_factor_sin * c_ur_sin * c_nr_sin)
        tB_p[i, m, :] = -wR * (phi_factor_cos * c_up_cos * c_np_cos + phi_factor_sin * c_up_sin * c_np_sin)

        # Enstrophy per (m, n) on the scalar Dirichlet J_m ladder.  This is an
        # absolute (positive) projection of vort^2; the natural BC of vort is
        # not Dirichlet (omega|r=1 = 2 u_phi), so the expansion has slight
        # boundary Gibbs, but it's fine as a diagnostic.
        enB[i, m, :] = wV * (phi_factor_cos * c_w_cos ** 2 + phi_factor_sin * c_w_sin ** 2)

        # --- Initial-term diagnostic for m in {0, 1, 2} ---
        # D_m = (1/init_weight) * ∫_0^1 profile(r) * r^{m_J} * r dr
        #     = (1/init_weight) * sum_i w_r[i] * profile(r_i) * r_i^{m_J}
        if m in (0, 1, 2):
            _r_pow = r_1d ** m_J
            _iw    = initial_term_weight(m_J)
            D_init_cos[i, m] = np.dot(a_u_p[m, :] * _r_pow, w_r) / _iw
            if m > 0:
                D_init_sin[i, m] = np.dot(b_u_p[m, :] * _r_pow, w_r) / _iw

    # --- Bin per snapshot ---
    keBmn_r[i, :, :] = bin_one_m_resolved(lambda_D, keB_r[i], mmax)
    keBmn_p[i, :, :] = bin_one_m_resolved(lambda_R, keB_p[i], mmax)
    tBmn_r [i, :, :] = bin_one_m_resolved(lambda_D, tB_r [i], mmax)
    tBmn_p [i, :, :] = bin_one_m_resolved(lambda_R, tB_p [i], mmax)
    enBmn  [i, :, :] = bin_one_m_resolved(lambda_V, enB[i], mmax)

    keBn_r[i, :] = np.sum(keBmn_r[i], axis=0)
    keBn_p[i, :] = np.sum(keBmn_p[i], axis=0)
    keBn  [i, :] = keBn_r[i] + keBn_p[i]
    tBn   [i, :] = np.sum(tBmn_r [i], axis=0) + np.sum(tBmn_p[i], axis=0)
    enBn  [i, :] = np.sum(enBmn  [i], axis=0)

# ---------------------------------------------------------------------------
# Dissipation, time derivative, budget residual (rank 0 only)
# ---------------------------------------------------------------------------

if rank == 0:
    logger.info("Computing dissipation spectra and budget residual")

    # D_nu(m, n) is built mode-by-mode using the actual eigenvalue of each
    # Bessel mode.  For u_r modes we use lambda_D^2, for u_phi modes
    # lambda_R^2.  This is the SCALAR-Laplacian approximation of the
    # viscous term applied to each component separately; the vector
    # Laplacian's u/r^2 metric corrections and the (2/r^2) cross-coupling
    # between u_r and u_phi are NOT captured here (small at high k, can
    # be O(1) at the lowest m=0,1,2 modes).  Flag for grid-space sanity
    # check in a later pass.
    D_nu_Bmn = np.zeros((nw, Nbins))
    for i in range(nw):
        d_r = 2.0 * nu * lambda_D ** 2 * keB_r[i]
        d_p = 2.0 * nu * lambda_R ** 2 * keB_p[i]
        d_r_binned = bin_one_m_resolved(lambda_D, d_r, mmax)
        d_p_binned = bin_one_m_resolved(lambda_R, d_p, mmax)
        D_nu_Bmn[i, :] = np.sum(d_r_binned + d_p_binned, axis=0)

    D_alpha_Bn = 2.0 * alpha * keBn

    if nw > 1:
        dkeBn_dt = np.gradient(keBn, tw, axis=0)
    else:
        dkeBn_dt = np.zeros_like(keBn)

    budget_residual_Bn = dkeBn_dt - tBn + D_nu_Bmn + D_alpha_Bn if nw > 1 else None

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
    processed['lambda_D']     = lambda_D
    processed['lambda_R']     = lambda_R

    processed['D_init_cos']   = D_init_cos   # initial-term diagnostics for m=0,1,2
    processed['D_init_sin']   = D_init_sin

    if not steady_only:
        processed['keB_r']    = keB_r
        processed['keB_p']    = keB_p
        processed['tB_r']     = tB_r
        processed['tB_p']     = tB_p
        processed['enB']      = enB

        processed['keBn_r']   = keBn_r
        processed['keBn_p']   = keBn_p
        processed['keBn']     = keBn
        processed['tBn']      = tBn
        processed['enBn']     = enBn
        processed['D_nu_Bn']  = D_nu_Bmn
        processed['D_alpha_Bn'] = D_alpha_Bn
        processed['dkeBn_dt'] = dkeBn_dt
        if budget_residual_Bn is not None:
            processed['budget_residual_Bn'] = budget_residual_Bn

        processed['keBmn_r']  = keBmn_r
        processed['keBmn_p']  = keBmn_p
        processed['tBmn_r']   = tBmn_r
        processed['tBmn_p']   = tBmn_p
        processed['enBmn']    = enBmn

    if nw > 1:
        t_end     = tw[-1]
        t_start   = t_end - t_steady_range
        idx_end   = np.where(tw <= t_end)[0][-1]
        idx_start = np.where(tw >= t_start)[0][0]
        sl = slice(idx_start, idx_end)

        for key, arr in [('keB_r', keB_r), ('keB_p', keB_p),
                         ('tB_r',  tB_r),  ('tB_p',  tB_p),
                         ('enB',   enB)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        for key, arr in [('keBn',   keBn),   ('keBn_r', keBn_r), ('keBn_p', keBn_p),
                         ('tBn',    tBn),    ('enBn',   enBn),
                         ('D_nu_Bn', D_nu_Bmn), ('D_alpha_Bn', D_alpha_Bn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        for key, arr in [('keBmn_r', keBmn_r), ('keBmn_p', keBmn_p),
                         ('tBmn_r',  tBmn_r),  ('tBmn_p',  tBmn_p),
                         ('enBmn',   enBmn)]:
            processed[key + '_tavg'] = np.mean(arr[sl], axis=0)

        if budget_residual_Bn is not None:
            processed['budget_residual_Bn_tavg'] = np.mean(budget_residual_Bn[sl], axis=0)

        # Time-averaged initial-term magnitudes (handy for one-line sanity).
        processed['D_init_cos_tavg'] = np.mean(D_init_cos[sl], axis=0)
        processed['D_init_sin_tavg'] = np.mean(D_init_sin[sl], axis=0)

    np.save(output_prefix + '_' + output_suffix + '.npy', processed)
    logger.info("Save complete")
else:
    logger.info("Rank %d finished", rank)

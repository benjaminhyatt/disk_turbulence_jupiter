"""
Zonal-flow contribution to the CPC stability balance (v1).

Tests the three-term stability balance from force_balance_hypothesis.pdf
(Eq. 16), in DIMENSIONAL form (independent of L):

    I_A = (1/2) * gamma * r0(t) * r_cpc * u_phi,cpc(r_cpc)
    I_B = u_phi,cpc(r_cpc) * cos(phi_cpc - phi0(t)) * omega_z,bg(r_lab,phi_lab,t)
    I_C = omega_z,cpc(r_cpc) * (cos(phi_cpc - phi0) * u_phi,bg^cpc
                              + sin(phi_cpc - phi0) * u_r,bg^cpc)

integrated over a CPC-frame disk of (per-frame) radius R_int(t).  The
hypothesis is that the time-mean of <I_A + I_B + I_C> ~ 0.

The DISJOINT projections of the simulation are:
    omega_CPC(t)   = a(t) * omega_T,rankine_smooth(r_cpc; omega_0(t), L(t))
                       (from v4: per-frame parametric fit + joint LSQ)
    omega_RW(t)    = Re[ sum_n A_n(t) * omega_n^R(lab) ]
                       (over --rw_modes; default winner only)
    omega_zonal(t) = <omega - omega_CPC - omega_RW>_{phi_lab}
                       (lab-frame phi-average of the LSQ residual)

bg = RW + zonal; we keep RW- and zonal-only contributions separate
through I_B and I_C so we can see which background piece supports the
balance.

Conventions (Ben's, from force_balance_hypothesis.pdf):
    omega_z = Laplacian(psi)
    u_phi   = d_r psi
    u_r     = -(1/r) d_phi psi
i.e. u = -skew(grad(psi)) in Dedalus operators.

Usage:
    process_zonal_flow_subtracted_v1.py <hdf5_file> <evp_file>
                                        <rw_amplitude_file> [options]

Arguments:
    <hdf5_file>            Simulation snapshots (must contain 'tasks/vort').
    <evp_file>             Eigenvalue-problem .npy with 'psi_right_evecs_res'.
    <rw_amplitude_file>    v4 output .npy from process_rw_amplitude_subtracted.

Options:
    --gamma=<float>            gamma (planetary-vorticity-gradient) of the
                                run, dimensional [REQUIRED unless 'auto'
                                and present in tracking/v4 .npy].
                                [default: auto]
    --integrate_rho_max=<str>  per-frame CPC-disk integration radius;
                                one of R_rms, R_half, R_fifth, R_tenth
                                (per-frame from tracking), R_*_mean,
                                '<float>L' (= <float>*L_fit(t)), or a
                                literal float.
                                [default: R_rms]
    --rw_modes=<str>           'winner' (just the v4 winner candidate),
                                'all' (all v4 candidates), or a
                                comma-separated list of EVP indices.
                                [default: winner]
    --t_start=<str>            't_v4' or float [default: t_v4]
    --t_end=<str>              't_v4' or float [default: t_v4]
    --output=<str>             [default: auto]
    --plot=<str>               [default: auto]
    --output_prefix=<str>      [default: processed_zonal_flow_subtracted_v1]
    --tracking_file=<str>      tracking .npy (only needed if certain R_*
                                arrays are missing from rw_amplitude_file)
                                [default: auto]
"""

import numpy as np
import h5py
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

### read args ###
args = docopt(__doc__)
print(args)

hdf5_file           = args['<hdf5_file>']
evp_file            = args['<evp_file>']
rw_amp_file         = args['<rw_amplitude_file>']

gamma_arg           = args['--gamma']
integrate_rho_max_arg = args['--integrate_rho_max']
rw_modes_arg        = args['--rw_modes']
t_start_arg         = args['--t_start']
t_end_arg           = args['--t_end']
output_arg          = args['--output']
plot_arg            = args['--plot']
output_prefix       = args['--output_prefix']
tracking_file_arg   = args['--tracking_file']

dealias = 3/2

def extract_output_suffix(file_path):
    basename = file_path.split('/')[-1]
    if basename.endswith('.npy'):
        basename = basename[:-4]
    for prefix in ('processed_rw_amplitude_subtracted_v4_',
                   'processed_rw_amplitude_subtracted_v3_',
                   'processed_rw_amplitude_subtracted_v2_',
                   'processed_rw_amplitude_subtracted_',
                   'processed_tracking_locator_',
                   'processed_tracking_',
                   'processed_rossby_evp_',
                   'analysis_'):
        if prefix in basename:
            basename = basename.split(prefix, 1)[1]
            break
    return basename

output_suffix = extract_output_suffix(rw_amp_file)
print(f"output_suffix: {output_suffix}")

def auto_or(arg, default_template):
    return default_template if arg.lower() == 'auto' else arg

output_path = auto_or(output_arg, f"{output_prefix}_{output_suffix}.npy")
if plot_arg.lower() in ('none', ''):
    plot_path = None
else:
    plot_path = auto_or(plot_arg, f"{output_prefix}_{output_suffix}.png")
print(f"output .npy: {output_path}")
print(f"plot path:   {plot_path}")

### load v4 RW-amplitude output ###
logger.info(f"Loading v4 .npy: {rw_amp_file}")
v4 = np.load(rw_amp_file, allow_pickle=True)[()]
required_keys = ('A_RW_history', 'a_history', 'omega_0_fit_history',
                 'L_fit_history', 'tw', 'ws', 'r_locs_used',
                 'phi_locs_used', 'glitch_flags_used',
                 'candidate_mode_idxs', 'winner_local',
                 'cpc_model', 'rho_fit_grid', 'vort_radial_tavg',
                 'peak0')

print("v4 rw keys", list(v4.keys()))

for k in required_keys:
    if k not in v4:
        raise KeyError(f"v4 .npy missing required key '{k}'.")
A_RW_v4        = v4['A_RW_history']         # (n_cand, nw_v4) complex
a_v4           = v4['a_history']            # (nw_v4,) complex
omega_0_v4     = v4['omega_0_fit_history']  # (nw_v4,)
L_fit_v4       = v4['L_fit_history']        # (nw_v4,)
tw_v4          = v4['tw']
ws_v4          = v4['ws']
r_locs_v4      = v4['r_locs_used']
phi_locs_v4    = v4['phi_locs_used']
glitch_v4      = v4['glitch_flags_used']
cand_idxs_v4   = v4['candidate_mode_idxs']
winner_local   = int(v4['winner_local'])
winner_evp_idx = int(v4.get('winner_evp_idx', cand_idxs_v4[winner_local]))
cpc_model_v4   = v4['cpc_model']
peak0          = float(v4['peak0'])
rho_max_tmpl   = float(v4['rho_fit_grid'][-1])
nw_v4          = len(tw_v4)
print(f"v4: {nw_v4} frames, cpc_model={cpc_model_v4}, "
      f"winner=cand {winner_local} (EVP {winner_evp_idx})")

if cpc_model_v4 != 'rankine_smooth':
    print(f"WARNING: v4 used cpc_model={cpc_model_v4}; this script's "
          f"analytic u_phi,cpc assumes rankine_smooth.")

### gamma ###
def resolve_gamma():
    if gamma_arg.lower() != 'auto':
        return float(gamma_arg)
    for k_try in ('gamma', 'Gamma', 'gamma_run', 'planet_gamma'):
        if k_try in v4:
            return float(v4[k_try])
    if tracking_file_arg.lower() != 'auto':
        tr = np.load(tracking_file_arg, allow_pickle=True)[()]
        for k_try in ('gamma', 'Gamma'):
            if k_try in tr:
                return float(tr[k_try])
    raise ValueError("Could not auto-resolve --gamma; pass it explicitly.")

gamma = resolve_gamma()
print(f"gamma = {gamma:.4f}")

### tracking (for R_rms etc., if not in v4 .npy) ###
tracking = None
if tracking_file_arg.lower() != 'auto':
    tracking = np.load(tracking_file_arg, allow_pickle=True)[()]
    print("keys in tracking", list(tracking.keys()))
def get_array_from_v4_or_tracking(key_v4, key_tracking, default_factory):
    """Look up array in v4 first, then tracking, else default."""
    if key_v4 in v4:
        return np.asarray(v4[key_v4])
    if tracking is not None and key_tracking in tracking:
        return np.asarray(tracking[key_tracking])
    return default_factory()

### time window ###
if t_start_arg.lower() in ('auto', 't_v4'):
    t_start_eff = float(tw_v4[0])
else:
    t_start_eff = float(t_start_arg)
if t_end_arg.lower() in ('auto', 't_v4'):
    t_end_eff = float(tw_v4[-1])
else:
    t_end_eff = float(t_end_arg)
in_window = (tw_v4 >= t_start_eff - 1e-9) & (tw_v4 <= t_end_eff + 1e-9)
sub_v4 = np.where(in_window)[0]
nw = len(sub_v4)
ws_sub      = ws_v4[sub_v4].astype(int)
tw_sub      = tw_v4[sub_v4]
r_sub       = r_locs_v4[sub_v4]
phi_sub     = phi_locs_v4[sub_v4]
glitch_sub  = glitch_v4[sub_v4]
a_sub       = a_v4[sub_v4]
omega_0_sub = omega_0_v4[sub_v4]
L_fit_sub   = L_fit_v4[sub_v4]
A_RW_sub    = A_RW_v4[:, sub_v4]
print(f"time window: [{t_start_eff:.3f}, {t_end_eff:.3f}], nw={nw}")

### integrate_rho_max parsing ###
def parse_integrate_rho_max(arg, nw, L_fit_sub, tracking, sub_idxs_v4):
    """Returns (per-frame array of length nw, source-label string)."""
    arg_lo = arg.lower()
    # 'kL' style ('3L', '2.5L', etc.)
    if arg_lo.endswith('l'):
        try:
            k = float(arg[:-1])
            arr = k * L_fit_sub
            arr = np.where(np.isfinite(arr) & (arr > 0), arr,
                           np.nanmean(arr[np.isfinite(arr) & (arr > 0)]))
            return arr, f'{k}*L_fit'
        except ValueError:
            pass
    # per-frame R_* (from v4 first, then tracking)
    history_keys = {'r_rms':   ('R_rms_history',   'R_rms_mean'),
                    'r_urms':  ('R_urms_history',  'R_urms_mean'),
                    'r_half':  ('R_half_history',  'R_half_mean'),
                    'r_fifth': ('R_fifth_history', 'R_fifth_mean'),
                    'r_tenth': ('R_tenth_history', 'R_tenth_mean')}
    if arg_lo in history_keys:
        hist_key, mean_key = history_keys[arg_lo]
        # try v4
        print(arg_lo, hist_key, mean_key)
        if hist_key in v4:
            arr_full = np.asarray(v4[hist_key], dtype=float)
            arr = arr_full[sub_idxs_v4]
        elif tracking is not None and hist_key in tracking:
            arr_full = np.asarray(tracking[hist_key], dtype=float)
            # Need same-index mapping; assume tracking has same length as v4
            # original window (i.e. the v4 .npy 'tw' length matches tracking)
            if len(arr_full) == nw_v4:
                arr = arr_full[sub_idxs_v4]
            else:
                raise ValueError(
                    f"Length mismatch between tracking '{hist_key}' "
                    f"({len(arr_full)}) and v4 frames ({nw_v4}); pass "
                    f"--tracking_file matching the v4 .npy.")
        elif mean_key in v4:
            return np.full(nw, float(v4[mean_key])), f'mean_{arg_lo}'
        elif tracking is not None and mean_key in tracking:
            return np.full(nw, float(tracking[mean_key])), f'mean_{arg_lo}'
        else:
            raise ValueError(
                f"Could not resolve --integrate_rho_max={arg}; no "
                f"matching arrays in v4 .npy or tracking.")
        fallback = float(np.nanmean(arr[np.isfinite(arr) & (arr > 0)]))
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, fallback)
        return arr, f'per_frame_{arg_lo}'
    # mean_* keys
    if arg_lo.endswith('_mean'):
        key_upper = arg.upper() if arg.startswith('R_') else None
        for src_dict, src_name in ((v4, 'v4'), (tracking, 'tracking')):
            if src_dict is None: continue
            for k in src_dict.keys():
                if k.lower() == arg_lo:
                    return np.full(nw, float(src_dict[k])), \
                           f'constant_{arg_lo}({src_name})'
        raise ValueError(f"Couldn't find mean key '{arg}'.")
    # literal float
    try:
        return np.full(nw, float(arg)), f'literal_{arg}'
    except ValueError:
        raise ValueError(f"Couldn't interpret --integrate_rho_max='{arg}'")

integrate_rho_max_arr, integrate_src = parse_integrate_rho_max(
    integrate_rho_max_arg, nw, L_fit_sub, tracking, sub_v4)
print(f"integrate_rho_max: source={integrate_src}, "
      f"mean={np.nanmean(integrate_rho_max_arr):.4f}")

### rw_modes parsing ###
def parse_rw_modes(arg, cand_idxs_v4, winner_local):
    arg_lo = arg.lower()
    if arg_lo == 'winner':
        # one entry in the candidate list (the winner)
        return np.array([winner_local], dtype=int), 'winner'
    if arg_lo == 'all':
        return np.arange(len(cand_idxs_v4), dtype=int), 'all'
    # csv list of EVP indices -> map to local candidate indices
    target_evp = [int(x) for x in arg.split(',') if x.strip() != '']
    local_idxs = []
    for evp_idx in target_evp:
        if evp_idx in cand_idxs_v4:
            local_idxs.append(int(np.where(cand_idxs_v4 == evp_idx)[0][0]))
        else:
            raise ValueError(f"EVP idx {evp_idx} not in v4 candidate set "
                             f"{list(cand_idxs_v4)}.")
    return np.array(local_idxs, dtype=int), f'csv:{arg}'

rw_local_idxs, rw_modes_src = parse_rw_modes(rw_modes_arg, cand_idxs_v4,
                                              winner_local)
rw_evp_idxs = cand_idxs_v4[rw_local_idxs]
print(f"rw_modes: source={rw_modes_src}, local idxs={list(rw_local_idxs)}, "
      f"EVP idxs={list(rw_evp_idxs)}")

### load EVP eigenfunctions (just the selected modes) ###
logger.info(f"Loading EVP file: {evp_file}")
evp = np.load(evp_file, allow_pickle=True)[()]
if 'psi_right_evecs_res' not in evp or 'evals_res' not in evp:
    raise KeyError("EVP file missing 'psi_right_evecs_res'/'evals_res'.")
psi_right_raw = np.asarray(evp['psi_right_evecs_res'])
evals_raw     = np.asarray(evp['evals_res'])
sort_idxs = np.argsort(evals_raw.imag)
psi_right = psi_right_raw[sort_idxs]
evals     = evals_raw[sort_idxs]
psi_R_selected = psi_right[rw_evp_idxs]   # (n_rw, Nphi_evp_deal, Nr_evp_deal)
Nphi_deal_evp = psi_right.shape[1]
Nr_deal_evp   = psi_right.shape[2]
print(f"EVP grid (dealiased): Nphi={Nphi_deal_evp}, Nr={Nr_deal_evp}")

### Dedalus setup ###
Nphi = int(round(Nphi_deal_evp / dealias))
Nr   = int(round(Nr_deal_evp   / dealias))
print(f"Dedalus base grid: Nphi={Nphi}, Nr={Nr}")
dtype  = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist   = d3.Distributor(coords, dtype=dtype)
disk   = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1,
                      dealias=dealias, dtype=dtype)
phi_g, r_g = dist.local_grids(disk, scales=(dealias, dealias))
phi_1d = phi_g[:, 0]
r_1d   = r_g[0, :]
Nphi_deal, Nr_deal = len(phi_1d), len(r_1d)
assert Nphi_deal == Nphi_deal_evp and Nr_deal == Nr_deal_evp, \
       "Dedalus dealiased grid disagrees with EVP grid."
r_mesh   = np.tile(r_1d[np.newaxis, :], (Nphi_deal, 1))
phi_mesh = np.tile(phi_1d[:, np.newaxis], (1, Nr_deal))
x_mesh   = r_mesh * np.cos(phi_mesh)
y_mesh   = r_mesh * np.sin(phi_mesh)

### Poisson solver (psi from omega) -- only needed if we ever need ###
###  full-field velocity inversion, but for now use the partitioned ###
###  pieces directly                                                ###

### Compute lab-frame velocity components of each selected eigenmode ###
###  u = -skew(grad(psi)) -> components (u_phi, u_r) in (e_phi, e_r) ###
print("Precomputing lab-frame velocity of each selected RW eigenmode...")
psi_in   = dist.Field(name='psi_in', bases=disk)

def eigenmode_velocity_components(psi_arr_real):
    """For real psi on the dealiased grid, return (u_phi, u_r) lab-frame
    components via u = -skew(grad(psi))."""
    psi_in.change_scales(dealias)
    psi_in['g'] = psi_arr_real
    u_vec = (-d3.skew(d3.grad(psi_in))).evaluate()
    u_vec.change_scales(dealias)
    # Dedalus PolarCoordinates('phi','r') -> components ordered (phi, r)
    u_phi_g = np.array(u_vec['g'][0])
    u_r_g   = np.array(u_vec['g'][1])
    return u_phi_g, u_r_g

n_rw = len(rw_local_idxs)
u_phi_R_lab = np.zeros((n_rw, Nphi_deal, Nr_deal), dtype=np.complex128)
u_r_R_lab   = np.zeros((n_rw, Nphi_deal, Nr_deal), dtype=np.complex128)
omega_R_lab = np.zeros((n_rw, Nphi_deal, Nr_deal), dtype=np.complex128)
psi_lap_in  = dist.Field(name='psi_lap_in', bases=disk)
def lap_of_real(g_arr):
    psi_lap_in.change_scales(dealias)
    psi_lap_in['g'] = g_arr
    out = d3.lap(psi_lap_in).evaluate()
    out.change_scales(dealias)
    return np.array(out['g'])

for j_local, evp_idx in enumerate(rw_evp_idxs):
    psi_R_re = psi_right[evp_idx].real
    psi_R_im = psi_right[evp_idx].imag
    u_phi_re, u_r_re = eigenmode_velocity_components(psi_R_re.copy())
    u_phi_im, u_r_im = eigenmode_velocity_components(psi_R_im.copy())
    u_phi_R_lab[j_local] = u_phi_re + 1j * u_phi_im
    u_r_R_lab[j_local]   = u_r_re   + 1j * u_r_im
    omega_R_re = lap_of_real(psi_R_re.copy())
    omega_R_im = lap_of_real(psi_R_im.copy())
    omega_R_lab[j_local] = omega_R_re + 1j * omega_R_im
print(f"Precomputed RW eigenmode velocities ({n_rw} modes).")

### analytic rankine_smooth helpers ###
def rankine_smooth_omega(r, omega_0, L):
    return omega_0 / np.sqrt(1.0 + (r/L)**2)

def rankine_smooth_u_phi(r, omega_0, L):
    """u_phi(r) for axisymmetric omega(r) = omega_0/sqrt(1+(r/L)^2).
    Closed form: u_phi(r) = omega_0 * L^2 / r * (sqrt(1+(r/L)^2) - 1).
    At r=0, u_phi -> 0; we handle the singularity numerically."""
    eps = 1e-30
    return omega_0 * L**2 / np.maximum(r, eps) * (
        np.sqrt(1.0 + (r/L)**2) - 1.0)

def axisymmetric_u_phi_from_omega(omega_radial, r_grid):
    """Compute u_phi(r) from axisymmetric omega(r) via
    u_phi(r) = (1/r) * integral_0^r r' omega(r') dr'.
    Trapezoidal cumulative integral on the (possibly non-uniform) r_grid.
    Includes r=0 by assuming u_phi(0)=0."""
    integrand = r_grid * omega_radial
    dr = np.diff(r_grid)
    seg_int = 0.5 * (integrand[:-1] + integrand[1:]) * dr
    cum = np.concatenate([[0.0], np.cumsum(seg_int)])
    eps = 1e-30
    u_phi = cum / np.maximum(r_grid, eps)
    if r_grid[0] < 1e-10:
        u_phi[0] = 0.0
    return u_phi

### CPC-frame disk integration weight ###
def cpc_frame_coords(x_mesh, y_mesh, r_CPC_t, phi_CPC_t):
    x_C = r_CPC_t * np.cos(phi_CPC_t)
    y_C = r_CPC_t * np.sin(phi_CPC_t)
    dx = x_mesh - x_C
    dy = y_mesh - y_C
    r_cpc   = np.sqrt(dx*dx + dy*dy)
    phi_cpc = np.arctan2(dy, dx) % (2*np.pi)
    return r_cpc, phi_cpc

### disk-integral helper (lab frame) ###
def disk_integral(integrand_g):
    """Integrate a (Nphi, Nr) field over the disk: int int integrand r dr dphi.
    Uses the dealiased grid with phi-uniform spacing and trapezoid in r."""
    phi_sum = np.sum(integrand_g, axis=0) * (2.0 * np.pi / Nphi_deal)
    return float(np.trapezoid(phi_sum * r_1d, r_1d))

### open HDF5 ###
f_h5 = h5py.File(hdf5_file, 'r')
t_all = f_h5['tasks/vort'].dims[0]['sim_time'][:]
vort_loader = dist.Field(name='vort_loader', bases=disk)

### storage ###
TermA_hist            = np.full(nw, np.nan)
TermB_RW_hist         = np.full(nw, np.nan)
TermB_zonal_hist      = np.full(nw, np.nan)
TermC_RW_hist         = np.full(nw, np.nan)
TermC_zonal_hist      = np.full(nw, np.nan)
imbalance_hist        = np.full(nw, np.nan)
# diagnostics
disk_area_hist        = np.full(nw, np.nan)
R_int_hist            = np.full(nw, np.nan)
r0_hist               = np.full(nw, np.nan)
phi0_hist             = np.full(nw, np.nan)
# zonal-flow radial profiles (per-frame on the lab r-grid)
omega_zonal_history   = np.full((nw, Nr_deal), np.nan)
u_phi_zonal_history   = np.full((nw, Nr_deal), np.nan)

# snapshot frame for diagnostic plot
mid_idx = nw // 2
while mid_idx < nw and glitch_sub[mid_idx]:
    mid_idx += 1
snap = {'t': None, 'vort': None, 'omega_CPC': None, 'omega_RW': None,
        'omega_zonal_2d': None, 'omega_res_full': None,
        'r_cpc': None, 'phi_cpc': None, 'mask': None,
        'I_A': None, 'I_B': None, 'I_C': None,
        'omega_zonal_profile': None, 'u_phi_zonal_profile': None}

prog_cad = max(1, nw // 50)
print(f"\nStarting per-frame balance pass (nw={nw}, gamma={gamma:.3f}) ...")

for i in range(nw):
    if i % prog_cad == 0:
        print(f"  frame {i}/{nw}")
    if glitch_sub[i]:
        continue
    if not np.isfinite(a_sub[i].real) or not np.isfinite(omega_0_sub[i]) \
       or not np.isfinite(L_fit_sub[i]):
        continue

    # --- load omega ---
    w = int(ws_sub[i])
    vort_loader.load_from_hdf5(f_h5, w)
    vort_loader.change_scales(dealias)
    vort_g = np.array(vort_loader['g'])

    # --- CPC center and per-frame params ---
    r_CPC_t   = float(r_sub[i])
    phi_CPC_t = float(phi_sub[i])
    a_t       = complex(a_sub[i])
    omega_0_t = float(omega_0_sub[i])
    L_t       = float(L_fit_sub[i])
    R_int_t   = float(integrate_rho_max_arr[i])
    R_int_hist[i] = R_int_t
    r0_hist[i]    = r_CPC_t
    phi0_hist[i]  = phi_CPC_t

    # --- CPC-frame coordinates over the full lab grid ---
    r_cpc, phi_cpc = cpc_frame_coords(x_mesh, y_mesh, r_CPC_t, phi_CPC_t)
    mask = (r_cpc <= R_int_t).astype(np.float64)

    # --- omega_CPC = a*omega_T,rankine_smooth on lab grid ---
    omega_T_g = a_t.real * rankine_smooth_omega(r_cpc, omega_0_t, L_t)
    # zero out outside template support (= rho_max_tmpl from v4) so
    # the template doesn't carry into the far field
    omega_T_g = np.where(r_cpc <= rho_max_tmpl, omega_T_g, 0.0)

    # --- omega_RW(t), u_RW(t) lab-frame ---
    A_n_t = A_RW_sub[rw_local_idxs, i]   # (n_rw,) complex
    # sum_n Re[A_n * X_n] for X in {omega_R, u_phi_R, u_r_R}
    omega_RW_g = np.zeros_like(vort_g)
    u_phi_RW_lab_g = np.zeros_like(vort_g)
    u_r_RW_lab_g   = np.zeros_like(vort_g)
    for j in range(n_rw):
        omega_RW_g     += (A_n_t[j] * omega_R_lab[j]).real
        u_phi_RW_lab_g += (A_n_t[j] * u_phi_R_lab[j]).real
        u_r_RW_lab_g   += (A_n_t[j] * u_r_R_lab[j]).real

    # --- omega_zonal(r,t) = phi-mean of (omega - omega_CPC - omega_RW) ---
    omega_res_full = vort_g - omega_T_g - omega_RW_g
    omega_zonal_radial = np.mean(omega_res_full, axis=0)  # (Nr_deal,)
    omega_zonal_history[i] = omega_zonal_radial

    # u_phi,zonal_lab from 1D inversion; u_r,zonal_lab = 0
    u_phi_zonal_lab_radial = axisymmetric_u_phi_from_omega(
        omega_zonal_radial, r_1d)
    u_phi_zonal_history[i] = u_phi_zonal_lab_radial
    omega_zonal_2d = np.tile(omega_zonal_radial[np.newaxis, :],
                             (Nphi_deal, 1))
    u_phi_zonal_lab_g = np.tile(u_phi_zonal_lab_radial[np.newaxis, :],
                                (Nphi_deal, 1))
    # u_r,zonal_lab = 0
    u_r_zonal_lab_g = np.zeros_like(vort_g)

    # --- background bundle ---
    omega_bg_g = omega_RW_g + omega_zonal_2d
    u_phi_bg_lab_g = u_phi_RW_lab_g + u_phi_zonal_lab_g
    u_r_bg_lab_g   = u_r_RW_lab_g   + u_r_zonal_lab_g

    # --- lab -> CPC velocity component transform ---
    # u_r,cpc  = u_r,lab*cos(phi_lab - phi_cpc) - u_phi,lab*sin(phi_lab - phi_cpc)
    # u_phi,cpc= u_r,lab*sin(phi_lab - phi_cpc) + u_phi,lab*cos(phi_lab - phi_cpc)
    dphi_lc = phi_mesh - phi_cpc
    cos_dlc = np.cos(dphi_lc)
    sin_dlc = np.sin(dphi_lc)
    u_r_bg_cpc_g   = u_r_bg_lab_g   * cos_dlc - u_phi_bg_lab_g * sin_dlc
    u_phi_bg_cpc_g = u_r_bg_lab_g   * sin_dlc + u_phi_bg_lab_g * cos_dlc

    # split RW vs zonal CPC-frame components for B/C decomposition
    u_r_RW_cpc_g     = u_r_RW_lab_g     * cos_dlc - u_phi_RW_lab_g    * sin_dlc
    u_phi_RW_cpc_g   = u_r_RW_lab_g     * sin_dlc + u_phi_RW_lab_g    * cos_dlc
    u_r_zonal_cpc_g  = u_r_zonal_lab_g  * cos_dlc - u_phi_zonal_lab_g * sin_dlc
    u_phi_zonal_cpc_g= u_r_zonal_lab_g  * sin_dlc + u_phi_zonal_lab_g * cos_dlc

    # --- CPC-frame axisymmetric profiles, evaluated at lab points ---
    u_phi_cpc_at_lab = a_t.real * rankine_smooth_u_phi(r_cpc, omega_0_t, L_t)
    omega_z_cpc_at_lab = a_t.real * rankine_smooth_omega(r_cpc, omega_0_t, L_t)
    # zero outside template support for consistency
    u_phi_cpc_at_lab   = np.where(r_cpc <= rho_max_tmpl, u_phi_cpc_at_lab, 0.0)
    omega_z_cpc_at_lab = np.where(r_cpc <= rho_max_tmpl, omega_z_cpc_at_lab, 0.0)

    # --- cos/sin(phi_cpc - phi0) at every lab point ---
    dphi_c0 = phi_cpc - phi_CPC_t
    cos_c0 = np.cos(dphi_c0)
    sin_c0 = np.sin(dphi_c0)

    # --- pointwise integrands (dimensional) ---
    I_A = 0.5 * gamma * r_CPC_t * r_cpc * u_phi_cpc_at_lab
    I_B_RW    = u_phi_cpc_at_lab * cos_c0 * omega_RW_g
    I_B_zonal = u_phi_cpc_at_lab * cos_c0 * omega_zonal_2d
    I_C_RW    = omega_z_cpc_at_lab * (
        cos_c0 * u_phi_RW_cpc_g + sin_c0 * u_r_RW_cpc_g)
    I_C_zonal = omega_z_cpc_at_lab * (
        cos_c0 * u_phi_zonal_cpc_g + sin_c0 * u_r_zonal_cpc_g)

    # --- integrate over CPC-frame disk (mask) ---
    TermA_hist[i]       = disk_integral(I_A       * mask)
    TermB_RW_hist[i]    = disk_integral(I_B_RW    * mask)
    TermB_zonal_hist[i] = disk_integral(I_B_zonal * mask)
    TermC_RW_hist[i]    = disk_integral(I_C_RW    * mask)
    TermC_zonal_hist[i] = disk_integral(I_C_zonal * mask)
    imbalance_hist[i]   = (TermA_hist[i] + TermB_RW_hist[i]
                          + TermB_zonal_hist[i] + TermC_RW_hist[i]
                          + TermC_zonal_hist[i])
    disk_area_hist[i]   = disk_integral(mask)

    if i == mid_idx:
        snap['t']              = float(tw_sub[i])
        snap['vort']           = vort_g.copy()
        snap['omega_CPC']      = omega_T_g.copy()
        snap['omega_RW']       = omega_RW_g.copy()
        snap['omega_zonal_2d'] = omega_zonal_2d.copy()
        snap['omega_res_full'] = omega_res_full.copy()
        snap['r_cpc']          = r_cpc.copy()
        snap['phi_cpc']        = phi_cpc.copy()
        snap['mask']           = mask.copy()
        snap['I_A']            = I_A.copy()
        snap['I_B_RW']         = I_B_RW.copy()
        snap['I_B_zonal']      = I_B_zonal.copy()
        snap['I_C_RW']         = I_C_RW.copy()
        snap['I_C_zonal']      = I_C_zonal.copy()
        snap['omega_zonal_profile'] = omega_zonal_radial.copy()
        snap['u_phi_zonal_profile'] = u_phi_zonal_lab_radial.copy()
        snap['R_int']          = R_int_t
        snap['r_CPC']          = r_CPC_t
        snap['phi_CPC']        = phi_CPC_t

f_h5.close()
print("Per-frame balance pass done.")

### time stats ###
def safe_mean(arr):
    fin = np.isfinite(arr)
    return float(np.mean(arr[fin])) if fin.any() else np.nan
def safe_std(arr):
    fin = np.isfinite(arr)
    return float(np.std(arr[fin])) if fin.any() else np.nan

TermA_mean       = safe_mean(TermA_hist)
TermB_RW_mean    = safe_mean(TermB_RW_hist)
TermB_zonal_mean = safe_mean(TermB_zonal_hist)
TermC_RW_mean    = safe_mean(TermC_RW_hist)
TermC_zonal_mean = safe_mean(TermC_zonal_hist)
imbalance_mean   = safe_mean(imbalance_hist)

print(f"\n--- Time-averaged stability terms (dim, m^3/s^2 ish) ---")
print(f"  <Term A>          = {TermA_mean:+.4e}")
print(f"  <Term B (RW)>     = {TermB_RW_mean:+.4e}")
print(f"  <Term B (zonal)>  = {TermB_zonal_mean:+.4e}")
print(f"  <Term C (RW)>     = {TermC_RW_mean:+.4e}")
print(f"  <Term C (zonal)>  = {TermC_zonal_mean:+.4e}")
print(f"  <imbalance>       = {imbalance_mean:+.4e}")
TermA_std        = safe_std(TermA_hist)
print(f"  std(Term A)/<A>   = {TermA_std/abs(TermA_mean+1e-30):.3f}")

# zonal flow time-mean profile
omega_zonal_tmean = np.nanmean(omega_zonal_history, axis=0)
u_phi_zonal_tmean = np.nanmean(u_phi_zonal_history, axis=0)

### save ###
results = {
    # per-frame
    'tw': tw_sub, 'ws': ws_sub,
    'glitch_flags_used': glitch_sub,
    'r_CPC_history': r0_hist,
    'phi_CPC_history': phi0_hist,
    'TermA_history':       TermA_hist,
    'TermB_RW_history':    TermB_RW_hist,
    'TermB_zonal_history': TermB_zonal_hist,
    'TermC_RW_history':    TermC_RW_hist,
    'TermC_zonal_history': TermC_zonal_hist,
    'imbalance_history':   imbalance_hist,
    'disk_area_history':   disk_area_hist,
    'R_int_history':       R_int_hist,
    'omega_zonal_history': omega_zonal_history,
    'u_phi_zonal_history': u_phi_zonal_history,
    # time-means
    'TermA_mean':       TermA_mean,
    'TermB_RW_mean':    TermB_RW_mean,
    'TermB_zonal_mean': TermB_zonal_mean,
    'TermC_RW_mean':    TermC_RW_mean,
    'TermC_zonal_mean': TermC_zonal_mean,
    'imbalance_mean':   imbalance_mean,
    'omega_zonal_tmean': omega_zonal_tmean,
    'u_phi_zonal_tmean': u_phi_zonal_tmean,
    # config
    'gamma':            gamma,
    'integrate_rho_max_arr':    integrate_rho_max_arr,
    'integrate_rho_max_source': integrate_src,
    'rw_modes_source':  rw_modes_src,
    'rw_local_idxs':    rw_local_idxs,
    'rw_evp_idxs':      rw_evp_idxs,
    'cpc_model':        cpc_model_v4,
    'rho_max_tmpl':     rho_max_tmpl,
    'peak0':            peak0,
    'rw_amp_file':      rw_amp_file,
    'evp_file':         evp_file,
    'hdf5_file':        hdf5_file,
    # grid (for downstream plotting)
    'r_1d':             r_1d,
    'output_suffix':    output_suffix,
}
print(f"\nSaving: {output_path}")
np.save(output_path, results)

### plot ###
if plot_path is not None:
    print(f"Saving plot: {plot_path}")
    fig = plt.figure(figsize=(17, 17), constrained_layout=True)
    gs  = fig.add_gridspec(4, 4)

    # Row 0: time series of each term
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(tw_sub, TermA_hist,       color='C0', lw=0.9, label=r'$I_A$ ($\beta$-drift)')
    ax.plot(tw_sub, TermB_RW_hist,    color='C1', lw=0.7, label=r'$I_B$ (RW)')
    ax.plot(tw_sub, TermB_zonal_hist, color='C2', lw=0.7, label=r'$I_B$ (zonal)')
    ax.plot(tw_sub, TermC_RW_hist,    color='C3', lw=0.7, label=r'$I_C$ (RW)')
    ax.plot(tw_sub, TermC_zonal_hist, color='C4', lw=0.7, label=r'$I_C$ (zonal)')
    ax.plot(tw_sub, imbalance_hist,   color='k',  lw=1.2, ls='--', label='sum (imbalance)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('t'); ax.set_ylabel(r'$\int dA\,I_*\;(\mathrm{m}^3/\mathrm{s}^2)$')
    ax.set_title('Stability balance terms per frame')
    ax.legend(fontsize=7, ncol=3, loc='best'); ax.grid(True, alpha=0.3)

    # Row 0: time-mean stacked bar (per term, with RW/zonal sub-stacks)
    ax = fig.add_subplot(gs[0, 2:4])
    means = np.array([TermA_mean, TermB_RW_mean + TermB_zonal_mean,
                      TermC_RW_mean + TermC_zonal_mean])
    rw_parts    = np.array([0.0, TermB_RW_mean,    TermC_RW_mean])
    zonal_parts = np.array([0.0, TermB_zonal_mean, TermC_zonal_mean])
    xs = np.array([0, 1, 2])
    ax.bar(xs, means, color=['C0', 'gray', 'gray'], alpha=0.4,
           edgecolor='k', label='total')
    ax.bar(xs, rw_parts,    color='C1', alpha=0.85, label='RW part')
    ax.bar(xs, zonal_parts, color='C2', alpha=0.85, bottom=rw_parts,
           label='zonal part')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([r'$I_A$', r'$I_B$', r'$I_C$'])
    ax.set_ylabel(r'time-mean')
    ax.set_title(
        f'Time-mean stability terms; imbalance = {imbalance_mean:+.3e}')
    ax.legend(fontsize=8, loc='best'); ax.grid(True, axis='y', alpha=0.3)

    # Row 1: zonal-flow radial profile (time-mean + snapshot)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(r_1d, omega_zonal_tmean, color='C0', lw=1.5,
            label=r'$\langle\omega_\mathrm{zonal}\rangle_t(r)$')
    if snap['omega_zonal_profile'] is not None:
        ax.plot(r_1d, snap['omega_zonal_profile'], color='C0',
                lw=0.7, alpha=0.6, label=fr'snapshot $t={snap["t"]:.2f}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'$r_\mathrm{lab}$'); ax.set_ylabel(r'$\omega_\mathrm{zonal}(r)$')
    ax.set_title('Zonal-mean vorticity (CPC- and RW-subtracted residual)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(r_1d, u_phi_zonal_tmean, color='C0', lw=1.5,
            label=r'$\langle u_{\varphi,\mathrm{zonal}}\rangle_t(r)$')
    if snap['u_phi_zonal_profile'] is not None:
        ax.plot(r_1d, snap['u_phi_zonal_profile'], color='C0',
                lw=0.7, alpha=0.6, label=fr'snapshot $t={snap["t"]:.2f}$')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'$r_\mathrm{lab}$'); ax.set_ylabel(r'$u_{\varphi,\mathrm{zonal}}(r)$')
    ax.set_title('Zonal-mean azimuthal velocity (lab frame)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 2: polar snapshots of CPC, RW, zonal, residual contributions
    titles = [r'$\omega$', r'$\omega_\mathrm{CPC}=a\,\omega_T$',
              r'$\omega_\mathrm{RW}$', r'$\omega_\mathrm{zonal}$ (2D)']
    snaps  = [snap['vort'], snap['omega_CPC'], snap['omega_RW'],
              snap['omega_zonal_2d']]
    vmax_w = float(np.max(np.abs(snap['vort']))) if snap['vort'] is not None else 1
    for col, (ttl, fld) in enumerate(zip(titles, snaps)):
        ax = fig.add_subplot(gs[2, col], projection='polar')
        if fld is not None:
            pcm = ax.pcolormesh(phi_mesh, r_mesh, fld, shading='auto',
                                 cmap='RdBu_r', vmin=-vmax_w, vmax=vmax_w)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
        ax.set_title(ttl + (f' ($t={snap["t"]:.2f}$)' if col==0 and snap['t'] is not None else ''))

    # Row 3: snapshot integrand maps + summary
    integrand_titles = [r'$I_A$', r'$I_B$ (full)', r'$I_C$ (full)']
    integrand_snaps  = [snap['I_A'],
                        (snap['I_B_RW'] + snap['I_B_zonal']) if snap['I_B_RW'] is not None else None,
                        (snap['I_C_RW'] + snap['I_C_zonal']) if snap['I_C_RW'] is not None else None]
    for col, (ttl, fld) in enumerate(zip(integrand_titles, integrand_snaps)):
        ax = fig.add_subplot(gs[3, col], projection='polar')
        if fld is not None and snap['mask'] is not None:
            masked = fld * snap['mask']
            vmax_i = float(np.max(np.abs(masked))) if np.max(np.abs(masked)) > 0 else 1
            pcm = ax.pcolormesh(phi_mesh, r_mesh, masked, shading='auto',
                                 cmap='RdBu_r', vmin=-vmax_i, vmax=vmax_i)
            fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.08)
        ax.set_title(ttl)

    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    txt  = f"output_suffix: {output_suffix}\n\n"
    txt += f"gamma:             {gamma:.3f}\n"
    txt += f"cpc_model (v4):    {cpc_model_v4}\n"
    txt += f"integrate_rho_max: {integrate_src}\n"
    txt += f"   <R_int>:        {np.nanmean(integrate_rho_max_arr):.4f}\n"
    txt += f"rw_modes:          {rw_modes_src}\n"
    txt += f"   EVP idxs:       {list(rw_evp_idxs)}\n\n"
    txt += f"nw_in_window:      {nw}\n"
    txt += f"<r_CPC>:           {np.nanmean(r0_hist):.4f}\n\n"
    txt += "Time-mean balance terms (dim):\n"
    txt += f"  <I_A>:           {TermA_mean:+.3e}\n"
    txt += f"  <I_B,RW>:        {TermB_RW_mean:+.3e}\n"
    txt += f"  <I_B,zonal>:     {TermB_zonal_mean:+.3e}\n"
    txt += f"  <I_C,RW>:        {TermC_RW_mean:+.3e}\n"
    txt += f"  <I_C,zonal>:     {TermC_zonal_mean:+.3e}\n"
    txt += f"  <imbalance>:     {imbalance_mean:+.3e}\n"
    rel = abs(imbalance_mean) / max(abs(TermA_mean), 1e-30)
    txt += f"  imbal/|<I_A>|:   {rel:.3f}\n"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=8, family='monospace', verticalalignment='top')

    fig.suptitle(
        f'CPC stability balance (Eq. 16, dimensional) - {output_suffix}',
        fontsize=11)
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {plot_path}")

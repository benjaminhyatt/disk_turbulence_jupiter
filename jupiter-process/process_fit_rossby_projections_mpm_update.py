"""
Fit oscillation frequencies of Rossby mode projection amplitudes using the
Matrix Pencil Method (MPM), loading from pre-processed projection output.
 
Usage:
    process_fit_rossby_projections_mpm.py <processed_file> [options]
 
Options:
    --output=<str>              prefix for the output file [default: processed_rossby_projection_mpm]
    --K=<int>                   number of sinusoidal components to fit per mode [default: 8]
    --sv_thresh=<float>         min value a singular value must be above to be retained [default: 5e-1]
    --omega_close_rtol=<float>  relative tol for flagging a frequency as being close to that of dominant mode [default: 1e-1]
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from docopt import docopt
 
### read in options ###
args = docopt(__doc__)
logger.info("args read in")
print(args)

processed_file_str = args['<processed_file>']
output = args['--output']
K = int(args['--K'])
sv_thresh_opt = float(args['--sv_thresh'])
omega_close_rtol = float(args['--omega_close_rtol'])

output_prefix = output
output_suffix = processed_file_str.split('processed_rossby_projection_')[1].split('.')[0].split('/')[0]

### fitting functions ###
def matrix_pencil_frequency(y, dt, K_max=8, sv_thresh=sv_thresh_opt):
    """
    Estimate K frequencies via MPM (Matrix Pencil Method)
    """
    N = len(y)
    L = N // 2  # pencil parameter choice
 
    # build Hankel matrix of shape (N - L) x (L + 1)
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
 
    # SVD of Y
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    
    # retain values above threshold, ensuring even and not exceeding specified K_max
    s_norm = s / s[0]
    print(s_norm[:12])
    K_thresh = int(np.sum(s_norm > sv_thresh))
    if K_thresh < 2:
        print('svd did not yield any values larger than specified threshhold')
    K_thresh = np.max((2, K_thresh))
    if K_thresh % 2 != 0:
        print('odd K_thresh = %d, increasing by 1' %(K_thresh))
        K_thresh += 1
    K = np.min((K_thresh, K_max))
    print(K, K_thresh)
    # construct one-step offset sub-matrices 
    U1 = U[:, :2*K] 
    U1_top = U1[:-1, :]
    U1_bot = U1[1:, :]

    # least squares solve for Z (2K x 2K) such that U1_bot (approx)= U1_top @ Z
    Z = np.linalg.pinv(U1_top) @ U1_bot     # eigenvalue solve
    # extract info from Z
    print(2*K)
    poles = np.linalg.eigvals(Z)            # poles = z_k = exp(lambda_k * dt)
    print("poles", poles)
    log_poles = np.log(poles) / dt          # lambda_k
    print("log", log_poles, dt)
    freqs = log_poles.imag
    print("freqs", freqs)
    decays = log_poles.real                 
 
    # keep only positive parts of each conjugate pair
    pos_mask = freqs > 0
    if np.sum(pos_mask) == 0:
        print('warning: no postive parts found', freqs)
    return freqs[pos_mask], decays[pos_mask], s_norm, K
 
def fit_multicomp(t, y, omegas):
    cols = []
    for omega in omegas:
        cols += [np.sin(omega * t), np.cos(omega * t)]
    A_mat = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)

    amps, phases, fits = [], [], []
    for k, omega in enumerate(omegas):
        c1, c2 = coeffs[2*k], coeffs[2*k + 1]
        amps.append(np.sqrt(c1**2 + c2**2))
        phases.append(np.arctan2(c2, c1))
        fits.append(A_mat[:, 2*k:2*k + 2] @ coeffs[2*k: 2*k + 2])

    # sort by amp, greatest to least
    amp_order = np.argsort(amps)[::-1]
    return (np.array(amps)[amp_order],
            np.array(phases)[amp_order],
            omegas[amp_order],
            np.array(fits)[amp_order],
            A_mat @ coeffs)

### load processed projection file ###
logger.info("loading: " + processed_file_str)
processed = np.load(processed_file_str, allow_pickle=True)[()]
 
tw = processed['tw']
idxs_include = processed['idxs_include']
print("idxs_include", idxs_include)
projl2_track = processed['projl2']
evals_re = np.copy(processed['evals_re'])
evals_im = np.copy(processed['evals_im'])
drifts = np.copy(processed['drifts'])
 
# time sampling
nidxs = len(idxs_include)
nw = len(tw)
print("nw", nw)
dt = tw[1] - tw[0]

### keep track of fit outputs in first sweep (sizes may vary)
all_omegas = [None] * nidxs
all_decays = [None] * nidxs
all_s_norms = [None] * nidxs
all_amps = [None] * nidxs
all_phases = [None] * nidxs
all_component_fits = [None] * nidxs
all_total_fits = [None] * nidxs

### enter first pass ###
#logger.info('entering first-pass -- fitting loop')
print('entering first-pass -- fitting loop')
for i, idx in enumerate(idxs_include):
    #logger.info('idx = %d, eigenvalue = %f + i%f, drift = %e' % (idx, evals_re[idx], evals_im[idx], drifts[idx]))
    print('idx = %d, eigenvalue = %f + i%f, drift = %e' % (idx, evals_re[idx], evals_im[idx], drifts[idx])) 
    # projection data to fit for ith read-in eigenmode
    signal = projl2_track[i, :]
 
    # frequency estimation via MPM
    freqs, decays, s_norm, K_used = matrix_pencil_frequency(signal, dt, K_max=K)

    # skip to next i if MPM fails
    if len(freqs) == 0:
        print('i=%d, idx=%d: MPM returned no positive frequencies' % (i, idx))
        continue
 
    # least square fits (results are returned sorted by amp, greatest to least)
    amps, phases, omegas, component_fits, total_fit = fit_multicomp(tw, signal, freqs)
    
    # keep track
    all_omegas[i] = omegas
    all_decays[i] = decays
    all_s_norms[i] = s_norm[:2*K_used]
    all_amps[i] = amps
    all_phases[i] = phases
    all_component_fits[i] = component_fits
    all_total_fits[i] = total_fit

### identify dominant mode ###
# maximize over the largest amplitudes for each i, if not None (considers 0 if None)
dom_i = np.argmax([amps[0] if amps is not None else 0. for amps in all_amps])
omega_dom = all_omegas[dom_i][0]
print('dominant mode: i=%d, omega_dominant=%.4f' % (dom_i, omega_dom))

### allocate second-pass output arrays ###

# raw mpm outputs: all frequencies and decay rates returned per mode
# (intrinsic frequency choices are stored separately in projl2_fit_params below)
mpm_freqs = np.zeros((nidxs, K))
mpm_decays = np.zeros((nidxs, K))
mpm_s_norms = np.zeros((nidxs, int(2*K))) # K largest normalized singular values for each mode

# primary selections (prioritizes dominant amplitudes when presented with choice)
projl2_fit = np.zeros((nidxs, nw))
projl2_fit_params = np.zeros((nidxs, 3))       # (A, omega, phi)
projl2_rel_err = np.zeros((nidxs, nw))

# secondary selections (linear theory-based choices -- only populated in ambiguous cases)
projl2_fit_lt = np.zeros((nidxs, nw))
projl2_fit_params_lt = np.zeros((nidxs, 3))    # (A, omega, phi)
projl2_rel_err_lt = np.zeros((nidxs, nw))

# diagnostic flags
ambig_flag = np.zeros(nidxs, dtype=bool) # True if ambiguous secondary case encountered
agree_flag = np.zeros(nidxs, dtype=bool) # True if secondary criteria within ambig case agreed
lt_candidate_idx = np.full(nidxs, -1, dtype=int) # index into mpm_freqs[i] of the lt candidate (-1 if unambiguous)

### second-pass -- sort results based on suspected dominant mode
#logger.info('entering second-pass -- sorting loop')
print('entering second-pass -- sorting loop')
for i, idx in enumerate(idxs_include):
   
    if all_omegas[i] is None:
        print('i=%d, idx=%d: skipping -- no MPM result from first pass' % (i, idx))
        continue
 
    signal = projl2_track[i, :]
    # temporarily unpack pertinent tracked results from earlier
    omegas = all_omegas[i]
    amps = all_amps[i]
    phases = all_phases[i]
    component_fits = all_component_fits[i]

    # assign leading result
    if i == dom_i:
        omega_intrinsic = omegas[0]
        amp_intrinsic = amps[0]
        phase_intrinsic = phases[0]
        fit_intrinsic = component_fits[0]
        print('i=%d (dominant): omega_intrinsic=%.4f, A=%.4f' % (i, omega_intrinsic, amp_intrinsic))
    
    # determine the instrinsic result that is not redundant with dominant mode 
    else:

        # compare other modes (omegas, i.e., all of the identified freqs from this i in first pass) to dominant mode
        near_dom_mask  = np.abs(omegas - omega_dom) / omega_dom < omega_close_rtol
        remainder_mask = ~near_dom_mask

        # if no other frequencies could be found, fall back to the leading result
        # (might need to reduce omega_close_rtol or sv_thresh, or increase K_max)
        if np.sum(remainder_mask) == 0:
            omega_intrinsic = omegas[0]
            amp_intrinsic = amps[0]
            phase_intrinsic = phases[0]
            fit_intrinsic = component_fits[0]
            print('i=%d, idx=%d: no freq distinct from omega_dom, storing leading component' % (i, idx))
    
        # 'non-ambiguous' -- if there is one and only one other option
        elif np.sum(remainder_mask) == 1:
            j = np.where(remainder_mask)[0][0]
            omega_intrinsic = omegas[j]
            amp_intrinsic = amps[j]
            phase_intrinsic = phases[j]
            fit_intrinsic = component_fits[j]
            print('i=%d, idx=%d: unambiguous intrinsic omega=%.4f, A=%.4f' % (i, idx, omega_intrinsic, amp_intrinsic))
    
        # 'ambiguous' -- use other criteria to make a choice from several other options
        else:
            ambig_flag[i] = True

            remainder_omegas = omegas[remainder_mask]
            remainder_amps = amps[remainder_mask]
            remainder_phases = phases[remainder_mask]
            remainder_fits = component_fits[remainder_mask]
            remainder_idxs = np.where(remainder_mask)[0]

            # criterion 1: largest amplitude
            best_amp = np.argmax(remainder_amps)
            # criterion 2: closest to linear theory prediction
            omega_pred = evals_re[idx]
            best_pred = np.argmin(np.abs(remainder_omegas - omega_pred))
            
            # decision
            agree_flag[i] = (best_amp == best_pred)
            if agree_flag[i]:
                print('i=%d, idx=%d: ambiguous (%d candidates), both criteria agree on omega=%.4f' % (i, idx, len(remainder_omegas), remainder_omegas[best_amp]))
            else:
                print('i=%d, idx=%d: ambiguous (%d candidates), criteria disagree (amp -> %.4f, linear theory -> %.4f)' % (i, idx, len(remainder_omegas), remainder_omegas[best_amp], remainder_omegas[best_pred]))

            lt_candidate_idx[i] = remainder_idxs[best_pred]  # index into mpm_freqs[i]

            # primary selection: prefer amplitude
            j = remainder_idxs[best_amp]
            omega_intrinsic = omegas[j]
            amp_intrinsic = amps[j]
            phase_intrinsic = phases[j]
            fit_intrinsic = component_fits[j]

            # secondary selection: linear theory-based (stored separately for reference)
            j_lt = remainder_idxs[best_pred]
            omega_intrinsic_lt = omegas[j_lt]
            amp_intrinsic_lt = amps[j_lt]
            phase_intrinsic_lt = phases[j_lt]
            fit_intrinsic_lt = component_fits[j_lt]
    
            projl2_fit_lt[i, :] = fit_intrinsic_lt
            projl2_fit_params_lt[i, :] = np.array([amp_intrinsic_lt, omega_intrinsic_lt, phase_intrinsic_lt])
            projl2_rel_err_lt[i, :] = np.abs(signal - fit_intrinsic_lt) / np.abs(fit_intrinsic_lt)

    ### main results to store ###
    # 'instrinsic' selections
    projl2_fit[i, :] = fit_intrinsic
    projl2_fit_params[i, :] = np.array([amp_intrinsic, omega_intrinsic, phase_intrinsic])
    projl2_rel_err[i, :] = np.abs(signal - fit_intrinsic) / np.abs(fit_intrinsic)
    # all mpm outputs
    n_freqs = len(omegas)
    mpm_freqs[i, :n_freqs] = omegas
    mpm_decays[i, :n_freqs] = all_decays[i]
    mpm_s_norms[i, :len(all_s_norms[i])] = all_s_norms[i]

    print('stored: omega=%.4f, A=%.4f, phi=%.4f, mean_rel_err=%.4f' % (omega_intrinsic, amp_intrinsic, phase_intrinsic, np.mean(projl2_rel_err[i, :])))


### sort all results by eigenmodes' fit amplitudes ###
amp_order = np.argsort(projl2_fit_params[:, 0])[::-1]
print('amplitude-based sort order:', amp_order)

def reorder(arr, order, axis=0):
    return np.take(arr, order, axis=axis)

projl2_fit = reorder(projl2_fit, amp_order)
projl2_fit_params = reorder(projl2_fit_params, amp_order)
projl2_rel_err = reorder(projl2_rel_err, amp_order)
projl2_fit_lt = reorder(projl2_fit_lt, amp_order)
projl2_fit_params_lt = reorder(projl2_fit_params_lt, amp_order)
projl2_rel_err_lt = reorder(projl2_rel_err_lt, amp_order)
mpm_freqs = reorder(mpm_freqs, amp_order)
mpm_decays = reorder(mpm_decays, amp_order)
mpm_s_norms = reorder(mpm_s_norms, amp_order)
ambig_flag = reorder(ambig_flag, amp_order)
agree_flag = reorder(agree_flag, amp_order)
lt_candidate_idx = reorder(lt_candidate_idx, amp_order)
evals_re = reorder(evals_re, amp_order)
evals_im = reorder(evals_im, amp_order)
drifts = reorder(drifts, amp_order)

### assemble and save output ###
fit_results = {}
fit_results.update(processed)

# overwrite reordered data
fit_results['evals_re'] = evals_re # (reordered just above)
fit_results['evals_im'] = evals_im # (reordered just above)
fit_results['drifts'] = drifts # (reordered just above)
fit_results['projdot_c'] = reorder(processed['projdot_c'], amp_order)
fit_results['projdot_s'] = reorder(processed['projdot_s'], amp_order)
fit_results['projdot'] = reorder(processed['projdot'], amp_order)
fit_results['projdot_stats'] = reorder(processed['projdot_stats'], amp_order)
fit_results['projl2'] = reorder(processed['projl2'], amp_order)
fit_results['projl2_stats'] = reorder(processed['projl2_stats'], amp_order)
fit_results['ivpl2'] = reorder(processed['ivpl2'], amp_order)
fit_results['ivpl2_stats'] = reorder(processed['ivpl2_stats'], amp_order)
fit_results['evpl2'] = reorder(processed['evpl2'], amp_order)

# raw MPM output: fixed-size, zero-padded to K columns
fit_results['mpm_freqs'] = mpm_freqs
fit_results['mpm_decays'] = mpm_decays
fit_results['mpm_s_norms'] = mpm_s_norms
 
# primary selection: amplitude-based intrinsic frequency per mode
fit_results['projl2_fit'] = projl2_fit
fit_results['projl2_fit_params'] = projl2_fit_params    # (A, omega, phi)
fit_results['projl2_rel_err'] = projl2_rel_err
 
# secondary selection: linear theory-based (data present only if ambig_flag below is True)
fit_results['projl2_fit_lt'] = projl2_fit_lt
fit_results['projl2_fit_params_lt'] = projl2_fit_params_lt # (A, omega, phi)
fit_results['projl2_rel_err_lt'] = projl2_rel_err_lt
 
# diagnostics: check ambiguous_flag before interpreting agreement_flag or lt_candidate_idx
fit_results['ambig_flag'] = ambig_flag
fit_results['agree_flag'] = agree_flag
fit_results['lt_candidate_idx'] = lt_candidate_idx  # index into mpm_freqs[i] of lt candidate (-1 if unambiguous)
 
out_path = output_prefix + '_' + output_suffix + '.npy'
print("Saving output as:", out_path)
np.save(out_path, fit_results)

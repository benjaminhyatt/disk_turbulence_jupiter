"""
Fit oscillation frequencies of Rossby mode projection amplitudes using the
Matrix Pencil Method (MPM), loading from pre-processed projection output.
 
Usage:
    process_fit_rossby_projections_mpm.py <processed_file> [options]
 
Options:
    --output=<str>      prefix for the output file [default: processed_rossby_projection_mpm]
    --K=<int>           number of sinusoidal components to fit per mode [default: 3]
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

output_prefix = output
output_suffix = processed_file_str.split('processed_rossby_projection_')[1].split('.')[0].split('/')[0]

### fitting functions ###
def matrix_pencil_frequency(y, dt, K=1):
    """
    Estimate K frequencies via MPM (Matrix Pencil Method)
    """
    N = len(y)
    L = N // 2  # pencil parameter choice
 
    # build Hankel matrix of shape (N - L) x (L + 1)
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
 
    # perform SVD and truncate down to 2K components (one mode and its c.c.)
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    U1 = U[:, :2*K]
    
    # from U, get two sub matrices that are effectively a one timestep shift apart
    U1_top = U1[:-1, :]
    U1_bot = U1[1:, :]

    # least squares solve for Z (2K x 2K) such that U1_bot (approx)= U1_top @ Z
    Z = np.linalg.pinv(U1_top) @ U1_bot     # eigenvalue solve

    # extract info from Z
    poles = np.linalg.eigvals(Z)            # poles = z_k = exp(lambda_k * dt)
    #print("poles", poles)
    log_poles = np.log(poles) / dt          # lambda_k
    #print("log", log_poles, dt)
    freqs = log_poles.imag
    decays = log_poles.real                 
 
    # keep only positive parts of each conjugate pair
    pos_mask = freqs > 0
    if np.sum(pos_mask) == 0:
        print("warning: no postive parts found -- presumably getting zeros?", freqs)
    return freqs[pos_mask], decays[pos_mask]
 
 
def fit_amplitude_phase(t, y, omega):
    """
    For a given omega, find least squares A and phi for y = A * sin(omega * t + phi)
    """
    A_mat = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y)
    c1, c2 = coeffs
    A = np.sqrt(c1**2 + c2**2)
    phi = np.arctan2(c2, c1)
    return A, phi, A_mat @ coeffs

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
 
nidxs = len(idxs_include)
nw = len(tw)
print(nw)
dt = tw[1] - tw[0]

projl2_fit = np.zeros((nidxs, nw))
projl2_fit_params = np.zeros((nidxs, 3))
projl2_rel_err = np.zeros((nidxs, nw))
mpm_decays = np.zeros((nidxs, K))
mpm_freqs = np.zeros((nidxs, K))

### main fitting loop ###
logger.info('entering fitting loop')
for i, idx in enumerate(idxs_include):
    #logger.info('idx = %d, eigenvalue = %f + i%f, drift = %e' % (idx, evals_re[idx], evals_im[idx], drifts[idx]))
    print('idx = %d, eigenvalue = %f + i%f, drift = %e' % (idx, evals_re[idx], evals_im[idx], drifts[idx])) 
    signal = projl2_track[i, :]
 
    # frequency estimation via MPM
    freqs, decays = matrix_pencil_frequency(signal, dt, K=K)
    pos_mask = freqs > 0
    if np.sum(pos_mask) > 0:
        omega_mpm = freqs[0] # choosing to take the dominant component... may revisit 
        mpm_freqs[i, :len(freqs)] = freqs
        mpm_decays[i, :len(decays)] = decays
        #logger.info('MPM: omega = %.3f, freq = %.3f, decay = %.3e' % (omega_mpm, freqs[0], decays[0]))
        print('MPM: omega = %.3f, freq = %.3f, decay = %.3e' % (omega_mpm, freqs[0], decays[0])) 

        # amplitude and phase via least squares
        A_fit, phi_fit, fit = fit_amplitude_phase(tw, signal, omega_mpm)

        rel_err = np.abs(signal - fit) / np.abs(fit)
        projl2_fit[i, :] = fit
        projl2_fit_params[i, :] = np.array([A_fit, omega_mpm, phi_fit])
        projl2_rel_err[i, :] = rel_err
        #logger.info('fit: A = %.3f, omega = %.3f, phi = %.3f, mean_rel_err = %.3f' % (A_fit, omega_mpm, phi_fit, np.mean(rel_err)))
        print('fit: A = %.3f, omega = %.3f, phi = %.3f, mean_rel_err = %.3f' % (A_fit, omega_mpm, phi_fit, np.mean(rel_err)))
    else:
        print('i=%d, idx=%d, MPM unsuccessful – mode is likely not well-resolved, consider discarding' %(i, idx))
        
# sort results by order of mode amplitude (largest to smallest)
amp_order = np.argsort(projl2_fit_params[:, 0])[::-1]

projl2_fit = projl2_fit[amp_order, :]
projl2_rel_err = projl2_rel_err[amp_order, :]
mpm_freqs = mpm_freqs[amp_order, :]
mpm_decays = mpm_decays[amp_order, :]
evals_re = evals_re[amp_order]
evals_im = evals_im[amp_order]
drifts = drifts[amp_order]
#evals_re = evals_re[idxs_include[amp_order]]
#evals_im = evals_im[idxs_include[amp_order]]
#drifts = drifts[idxs_include[amp_order]]
projl2_fit_params = projl2_fit_params[amp_order, :]

print(amp_order)

fit_results = {}
# pass through everything from the original processed file
fit_results.update(processed)
del fit_results['evals_re']
del fit_results['evals_im']
del fit_results['drifts']

fit_results['projl2_fit'] = projl2_fit
fit_results['projl2_fit_params'] = projl2_fit_params # arrangement: A, omega, phi
fit_results['projl2_rel_err'] = projl2_rel_err
fit_results['mpm_freqs'] = mpm_freqs
fit_results['mpm_decays'] = mpm_decays
fit_results['evals_re'] = evals_re
fit_results['evals_im'] = evals_im
fit_results['drifts'] = drifts


fit_results['projdot_c'] = fit_results['projdot_c'][amp_order, :]
fit_results['projdot_s'] = fit_results['projdot_s'][amp_order, :]
fit_results['projdot'] = fit_results['projdot'][amp_order, :]
fit_results['projdot_stats'] = fit_results['projdot_stats'][amp_order, :]
fit_results['projl2'] = fit_results['projl2'][amp_order, :]
fit_results['projl2_stats'] = fit_results['projl2_stats'][amp_order, :]
fit_results['ivpl2'] = fit_results['ivpl2'][amp_order, :]
fit_results['ivpl2_stats'] = fit_results['ivpl2_stats'][amp_order, :]
fit_results['evpl2'] = fit_results['evpl2'][amp_order]


out_path = output_prefix + '_' + output_suffix + '.npy'
print("Saving output as:", out_path)
np.save(out_path, fit_results)

"""
Project IVP output onto specified Rossby mode from corresponding EVP 

Usage:
    process_rossby_projection.py <ivp_file> <evp_file> [options]

Options:
    --output=<str>          prefix in the name of the output file [default: processed_rossby_projection]
    
    --sort_1=<str>          option 1 of how to sort the EVP results ('abs', 're', 'im', 'drift') [default: drift]
    --sort_2=<str>          option 2 of how to sort the EVP results ('inc', 'dec') [default: dec]
    --idxs_include=<tuple>  tuple of ints of mode idx to include in the results [default: 0,1,2,3,4]
    --include_all=<bool>    if True, will produce results for all processed modes, and idxs_include will be modified as such [default: False]

    --t_start=<float>       sim time to begin projecting IVP writes [default: 50.]
    --t_end=<float>         sim time to end the same [default: 100.]
"""
import dedalus.public as d3
import h5py
import numpy as np
from scipy.stats import norm
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from docopt import docopt

def m_map(m, Nphi, flag):
    m_in = np.array(m)
    if not m_in.shape:
        m_in = np.array([m]) 

    if flag == 're':
        m_out = 4 * m_in
        mask = m_out > Nphi - 2 
        m_out[mask] = Nphi - 2 - 4 * (m_in[mask] - int(Nphi/4))
        return m_out
    elif flag == 'co':
        m_out = 2 * m_in
        mask = m_in < 0
        m_out[mask] += Nphi + 1
        return m_out
    else: 
        print("Invalid argument", flag)
        raise

# to project a (IVP radial coefficients for fixed m and t) onto b (EVP radial coefficients)
#def projdot(a, b):
#    return (np.dot(a, b) / np.dot(b, b)) #* b
def projdot(a, b_re, b_im):
    num = np.dot(a, b_re) # Re(a @ b) = Re(a @ conj(b))
    den = np.dot(b_re, b_re) + np.dot(b_im, b_im) # b @ conj(b)
    return num/den


### read in options ###
args = docopt(__doc__)
logger.info("args read in")
print(args)

ivp_file_str = args['<ivp_file>']#[0]
evp_file_str = args['<evp_file>']#[0]
output = args['--output']
sort_1 = args['--sort_1']
sort_2 = args['--sort_2']
idxs_include = args['--idxs_include']
idxs_include = idxs_include.split(',')
idxs_include = tuple(int(s) for s in idxs_include)

include_all = eval(args['--include_all'])

t_start = float(args['--t_start'])
t_end = float(args['--t_end'])

### determine relevant parameters from (either) file_str ###
def str_to_float(a):
    first = float(a[0])
    try:
        sec = float(a[2]) # if str begins with format XdY
    except:
        sec = 0
    if a[-3] == 'p':
        sgn = 1
    else:
        sgn = -1
    exp = int(a[-2:])
    return (first + sec/10) * 10**(sgn * exp)

output_suffix = evp_file_str.split('processed_rossby_evp_')[1].split('.')[0].split('/')[0]
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
m = int(output_suffix.split('m_')[1].split('_')[0])
inviscid = int(output_suffix.split('inviscid_')[1].split('_')[0])

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
kf_str = output_suffix.split('kf_')[1].split('_')[0]

alpha_read = str_to_float(alpha_str)
gamma_read = str_to_float(gamma_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)
kf_read = str_to_float(kf_str)

alpha_vals = np.array((1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 1200, 1920, 2372, 2500, 3200))
eps_vals = np.array([1.0, 2.0])
nu_vals = np.array([2e-4, 8/90000, 8e-5, 4e-5, 2e-5])
kf_vals = np.array((10, 20, 30, 40, 80))

alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]
k_force = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]

output_prefix = output
output_prefix += '_sort_' + sort_1 + '_' + sort_2

m_idx_co_plus = m_map(m, Nphi, 'co')[0] # assumes the m read in is non-negative
m_idx_co_minus = m_map(-m, Nphi, 'co')[0]
m_idx_re_c = m_map(m, Nphi, 're')[0]
m_idx_re_s = m_idx_re_c + 1

### load evp file ###
logger.info("loading: " + evp_file_str)
processed_evp = np.load(evp_file_str, allow_pickle=True)[()]

# apply sorting logic to determine mode(s) to work with
evals = processed_evp['evals_res']
drifts = processed_evp['drifts_res']
if sort_1 == 'drift':
    sort_by = drifts
elif sort_1 == 'abs':
    sort_by = np.abs(evals)
elif sort_1 == 're':
    sort_by = evals.real
elif sort_1 == 'im':
    sort_by = evals.imag
if sort_2 == 'inc':
    sort_idxs = np.argsort(sort_by)
elif sort_2 == 'dec':
    sort_idxs = np.argsort(sort_by)[::-1]

evals_sorted = evals[sort_idxs]
drifts_sorted = drifts[sort_idxs]
evecs = processed_evp['vort_evecs_res']
evecs_sorted = evecs[sort_idxs, :]

### d3 setup ###
dealias = 3/2
coords = d3.PolarCoordinates('phi', 'r')
dtype_re = np.float64
dist_re = d3.Distributor(coords, dtype=dtype_re)
disk_re = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype_re)
edge_re = disk_re.edge
radial_basis_re = disk_re.radial_basis
vort = dist_re.Field(name='vort', bases=disk_re) # for loading from ivp analysis
phi_deal, r_deal = dist_re.local_grids(disk_re, scales=(dealias, dealias))

dtype_co = np.complex128
dist_co = d3.Distributor(coords, dtype=dtype_co)
disk_co = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dealias=dealias, dtype=dtype_co)
edge_co = disk_co.edge
radial_basis_co = disk_co.radial_basis
vort_evec = dist_co.Field(bases=disk_co) # for selected eigenmode from evp
vort_ivp_co = dist_co.Field(bases=disk_co) # if we decide to cast the ivp field to complex dtype

### load in ivp analysis file ###
f = h5py.File(ivp_file_str)
t = np.array(f['tasks/u'].dims[0]['sim_time'])
ws = np.arange(np.where(t <= t_start)[0][-1], np.where(t >= t_end)[0][0] + 1)
nw = len(ws)
tw = t[ws] # w=0 corresponds to t=0

### begin work ###
if include_all:
    idxs_include = np.arange(0, len(evals_sorted))
nidxs = len(idxs_include)

vort_m_c_re_track = np.zeros((nidxs, nw))
vort_m_s_re_track = np.zeros((nidxs, nw))
vort_m_c_im_track = np.zeros((nidxs, nw))
vort_m_s_im_track = np.zeros((nidxs, nw))

projdot_c_track = np.zeros((nidxs, nw))
projdot_s_track = np.zeros((nidxs, nw))
projdot_track = np.zeros((nidxs, nw))
projdot_stats = np.zeros((nidxs, 2))

projl2_track = np.zeros((nidxs, nw))
projl2_stats = np.zeros((nidxs, 2))

projl2norm_track = np.zeros((nidxs, nw))
projl2norm_stats = np.zeros((nidxs, 2))

ivpl2_track = np.zeros((nidxs, nw))
ivpl2_stats = np.zeros((nidxs, 2))

evpl2_track = np.zeros((nidxs, nw))

logger.info('entering work loop')
for i, idx in enumerate(idxs_include):
    logger.info('idx = %d, eigenvalue = %f + i%f, drift = %e' %(idx, evals_sorted[idx].real, evals_sorted[idx].imag, drifts_sorted[idx]))

    vort_evec.change_scales(dealias)
    vort_evec['g'] = np.copy(evecs_sorted[idx, :])

    # loop over writes
    prog_cad = 5 #32
    for j, w in enumerate(ws):
        if j % prog_cad == 0:
            logger.info("writes loop: j = %d out of %d" %(j, nw))

        vort.load_from_hdf5(f, w)
        vort.change_scales(1)
        
        # projections via dot product
        vort_evec_m_c_re = vort_evec['c'][m_idx_co_plus, :].real + vort_evec['c'][m_idx_co_minus, :].real
        vort_evec_m_s_re = vort_evec['c'][m_idx_co_plus, :].imag - vort_evec['c'][m_idx_co_minus, :].imag
        vort_evec_m_c_im = vort_evec['c'][m_idx_co_plus, :].imag + vort_evec['c'][m_idx_co_minus, :].imag
        vort_evec_m_s_im = - vort_evec['c'][m_idx_co_plus, :].real + vort_evec['c'][m_idx_co_minus, :].real
        vort_m_c_re_track[i, j] = np.sqrt(np.dot(vort_evec_m_c_re, vort_evec_m_c_re))
        vort_m_s_re_track[i, j] = np.sqrt(np.dot(vort_evec_m_s_re, vort_evec_m_s_re))
        vort_m_c_im_track[i, j] = np.sqrt(np.dot(vort_evec_m_c_im, vort_evec_m_c_im))
        vort_m_s_im_track[i, j] = np.sqrt(np.dot(vort_evec_m_s_im, vort_evec_m_s_im))
        projdot_c = projdot(vort['c'][m_idx_re_c, :], vort_evec_m_c_re, vort_evec_m_c_im)
        projdot_s = projdot(vort['c'][m_idx_re_s, :], vort_evec_m_s_re, vort_evec_m_s_im)
        projdot_c_track[i, j] = 0.5*projdot_c
        projdot_s_track[i, j] = 0.5*projdot_s
        projdot_track[i, j] = 0.5*(projdot_c + projdot_s) # 0.5 comes from integrating cos^2 and sin^2 = 1/2 +/- cos(2*...)

        # projects via l2 inner product in d3
        vort_ivp_co['c'] *= 0.
        vort_ivp_co.change_scales(1)
        vort_ivp_co['g'] += np.copy(vort['g'])

        projl2_num = d3.integ(vort_ivp_co*np.conj(vort_evec)).evaluate()['g'][0, 0].real
        projl2_den = d3.integ(vort_evec*np.conj(vort_evec)).evaluate()['g'][0, 0].real
        evpl2_track[i, j] = np.sqrt(projl2_den)
        projl2_track[i, j] = projl2_num / projl2_den
        projl2norm_track[i, j] = projl2_num / np.sqrt(projl2_den)
        ivpl2_track[i, j] = np.sqrt(d3.integ(vort_ivp_co*np.conj(vort_ivp_co)).evaluate()['g'][0, 0].real)

    projdot_stats[i, :] = norm.fit(projdot_track[i, :]) # (mean, stddev)
    projl2_stats[i, :] = norm.fit(projl2_track[i, :]) # (mean, stddev)
    projl2norm_stats[i, :] = norm.fit(projl2norm_track[i, :]) # (mean, stddev)
    ivpl2_stats[i, :] = norm.fit(ivpl2_track[i, :]) # (mean, stddev)


processed = {}

processed['nw'] = nw
processed['ws'] = ws
processed['tw'] = tw

processed['idxs_include'] = idxs_include
processed['evals_re'] = evals_sorted.real
processed['evals_im'] = evals_sorted.imag
processed['drifts'] = drifts_sorted

processed['vort_m_c_re_track'] = vort_m_c_re_track
processed['vort_m_s_re_track'] = vort_m_s_re_track
processed['vort_m_c_im_track'] = vort_m_c_im_track
processed['vort_m_s_im_track'] = vort_m_s_im_track


processed['projdot_c'] = projdot_c_track
processed['projdot_s'] = projdot_s_track
processed['projdot'] = projdot_track ### should be equivalent to projl2 below
processed['projdot_stats'] = projdot_stats
processed['projl2'] = projl2_track ### (vort_ivp_co, vort_evec) / ||vort_evec||^2
processed['projl2_stats'] = projl2_stats
processed['projl2norm'] = projl2norm_track ### (vort_ivp_co, vort_evec) / ||vort_evec||
processed['projl2norm_stats'] = projl2norm_stats
processed['ivpl2'] = ivpl2_track ### ||vort_ivp_co|| -- can get ||.||^2 if needed
processed['ivpl2_stats'] = ivpl2_stats
processed['evpl2'] = evpl2_track ### ||vort_evec|| -- can get ||.||^2 if needed

print("Saving output as:", output_prefix + '_' + output_suffix + '.npy')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

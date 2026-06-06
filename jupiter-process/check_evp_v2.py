import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# quickly loading in some of the new processed evp results
evp_file_str = 'processed_rossby_evp_m_1_inviscid_0_nu_2em04_gam_1d2ep03_kf_2d0ep01_Nphi_512_Nr_256_eps_1d0ep00_alpha_1d0em02_ring_0_restart_evolved_0_tau_mod_1_seed_31415926_safety_1d0em01_timestepper_SBDF2_bc_sf.npy'
logger.info("loading: " + evp_file_str)
processed_evp = np.load(evp_file_str, allow_pickle=True)[()]

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

Nphi = 512
Nr = 256
m = 1

m_idx_co_plus = m_map(m, Nphi, 'co')[0] # assumes the m read in is non-negative
m_idx_co_minus = m_map(-m, Nphi, 'co')[0]
m_idx_re_c = m_map(m, Nphi, 're')[0]
m_idx_re_s = m_idx_re_c + 1

psi_right_evecs = processed_evp['psi_right_evecs_res']
psi_mleft_evecs = processed_evp['psi_mleft_evecs_res']
vort_right_evecs = processed_evp['vort_right_evecs_res']
vort_mleft_evecs = processed_evp['vort_mleft_evecs_res']


# quick d3 setup
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
psi_right_evec = dist_co.Field(bases=disk_co) # for selected eigenmode from evp
psi_mleft_evec = dist_co.Field(bases=disk_co) # modified left counterpart of right eigenmode
vort_right_evec = dist_co.Field(bases=disk_co) # for selected eigenmode from evp
vort_mleft_evec = dist_co.Field(bases=disk_co) # modified left counterpart of right eigenmode
vort_ivp_co = dist_co.Field(bases=disk_co) # to cast the ivp field to complex dtype

# test inner products for biorthonormality 
idx0 = 0
idx1 = 1

# looking for inner product = 1
psi_right_evec.change_scales(dealias)
psi_mleft_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx0, :]
psi_mleft_evec['g'] = psi_mleft_evecs[idx0, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_mleft_evec)*psi_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
psi_np = np.conj(psi_mleft_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 1", psi_d3, psi_np)

psi_right_evec.change_scales(dealias)
psi_mleft_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx0, :]
psi_mleft_evec['g'] = psi_mleft_evecs[idx1, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_mleft_evec)*psi_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c'][m_idx_co_plus, :]
psi_np = np.conj(psi_mleft_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 0", psi_d3, psi_np)

psi_right_evec.change_scales(dealias)
psi_mleft_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx1, :]
psi_mleft_evec['g'] = psi_mleft_evecs[idx0, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_mleft_evec)*psi_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
psi_np = np.conj(psi_mleft_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("(check for symmetry) looking for inner product = 0", psi_d3, psi_np)

# looking for inner product = 1
vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx0, :]
vort_mleft_evec['g'] = vort_mleft_evecs[idx0, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 1", vort_d3, vort_np)

vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx0, :]
vort_mleft_evec['g'] = vort_mleft_evecs[idx1, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 0", vort_d3, vort_np)

vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx1, :]
vort_mleft_evec['g'] = vort_mleft_evecs[idx0, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("(check for symmetry) looking for inner product = 0", vort_d3, vort_np)





### I am curious what right @ right looks like
# looking for inner product = 1
psi_right_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx0, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_right_evec)*psi_right_evec).evaluate()['g'][0, 0]
psi_np = np.conj(psi_right_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 1", psi_d3, psi_np)

psi_right_evec.change_scales(dealias)
psi_mleft_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx0, :]
psi_mleft_evec['g'] = psi_right_evecs[idx1, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_mleft_evec)*psi_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c'][m_idx_co_plus, :]
psi_np = np.conj(psi_mleft_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 0", psi_d3, psi_np)

psi_right_evec.change_scales(dealias)
psi_mleft_evec.change_scales(dealias)
psi_right_evec['g'] = psi_right_evecs[idx1, :]
psi_mleft_evec['g'] = psi_right_evecs[idx0, :]
psi_d3 = 0.5 * d3.Average(np.conj(psi_mleft_evec)*psi_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
psi_np = np.conj(psi_mleft_evec['c'][m_idx_co_plus, :]).T @ psi_right_evec['c'][m_idx_co_plus, :]
print("(check for symmetry) looking for inner product = 0", psi_d3, psi_np)

# looking for inner product = 1
vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx0, :]
vort_mleft_evec['g'] = vort_right_evecs[idx0, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 1", vort_d3, vort_np)

vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx0, :]
vort_mleft_evec['g'] = vort_right_evecs[idx1, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("looking for inner product = 0", vort_d3, vort_np)

vort_right_evec.change_scales(dealias)
vort_mleft_evec.change_scales(dealias)
vort_right_evec['g'] = vort_right_evecs[idx1, :]
vort_mleft_evec['g'] = vort_right_evecs[idx0, :]
vort_d3 = 0.5 * d3.Average(np.conj(vort_mleft_evec)*vort_right_evec).evaluate()['g'][0, 0]
#psi_np = np.conj(psi_mleft_evec['c']).T @ psi_right_evec['c']
vort_np = np.conj(vort_mleft_evec['c'][m_idx_co_plus, :]).T @ vort_right_evec['c'][m_idx_co_plus, :]
print("(check for symmetry) looking for inner product = 0", vort_d3, vort_np)



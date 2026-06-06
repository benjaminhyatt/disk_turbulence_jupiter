"""
Usage:
    process_rossby_proj_prep.py <ivp_file> <evp_file> <proj_file> [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_rossby_proj_prep]
"""
import numpy as np
import h5py
import dedalus.public as d3
from docopt import docopt
args = docopt(__doc__)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
### read arguments passed to script ###
print("args read in")
print(args)
file_str = args['<file>'][0]
output_prefix = args['--output']
ivp_file_str = args['<ivp_file>']#[0]
evp_file_str = args['<evp_file>']#[0]
proj_file_str = args['<proj_file>']

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


### read analysis file ###
logger.info("loading: " + ivp_file_str)
f = h5py.File(ivp_file_str)

### load evp file ###
logger.info("loading: " + evp_file_str)
processed_evp = np.load(evp_file_str, allow_pickle=True)[()]

### load proj file ###
logger.info("loading: " + proj_file_str)
processed_proj = np.load(proj_file_str, allow_pickle=True)[()]

# string parsing
output_suffix = ivp_file_str.split('analysis_')[1].split('.')[0].split('/')[0]
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
m_in = int(output_suffix.split('m_')[1].split('_')[0])
m_idx_c = m_map(m_in, Nphi, 're')

# d3 setup
dealias = 3/2
Nphi_deal = int(np.round(dealias * Nphi))
Nr_deal = int(np.round(dealias * Nr))
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
vort = dist.Field(name = 'vort', bases = disk)
vortm = dist.Field(name = 'vortm', bases = disk)

# polar to cartesian conversion
plot_func = lambda phi, r, data: (r*np.cos(phi), r*np.sin(phi), data)
Phi, R = plot_tools.quad_mesh(phi_deal[:, 0], r_deal[0, :])

### range of writes available from processed projections ###
ws = processed_proj['ws']
nw = processed_proj['nw']
tw = processed_proj['tw']

### organization for plotting ###
phis = np.zeros(nw)
Ztot = np.zeros((nw, Nphi_deal, Nr_deal))
Zm = np.zeros((nw, Nphi_deal, Nr_deal)) 
Zproj = np.zeros((nw, Nphi_deal, Nr_deal))

# Loop over writes
prog_cad = 32
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        logger.info("writes loop: i = %d out of %d" %(i, nw))

    # load in vorticity data from corresponding write
    vort.load_from_hdf5(f, w)
    vort.change_scales(dealias)
    Ztot[i, :, :] = np.copy(vort['g'])

    # m=m_in component of solution
    vortm['c'] *= 0.
    vortcm = np.copy(vort['c'][m_idx_c:m_idx_c+2, :])
    vortm['c'][m_idx_c:m_idx_c+2, :] += vortcm
    vortm.change_scales(dealias)
    Zm[i, :, :] = np.copy(vortm['g'])

    # We could loop over multiple evp proj here, and consider Zproj as their superposition
    # Note: also need to worry about getting the phase of the mode right, not just the amplitude



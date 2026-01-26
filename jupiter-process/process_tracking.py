"""
Usage:
    process_tracking.py <file>... [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_tracking]
    --t_out_start=<float>   sim time to begin tracking [default: 0.3]
    --t_out_end=<float>     sim time to stop tracking [default: 3.7]

    --use_cutoff=<bool>     flag True to ignore grid data greater than a specified radius [default: True]
    --use_stddev=<bool>     flag True to ignore grid data of size less than a multiple of the standard deviation [default: False]
    --track_all=<bool>      flag True to track all vorticity extrema (recommend using both options above) [default: False]
    --r_cutoff=<float>      provide a specified cutoff radius, if None, default will base cutoff on Lgamma [default: None]

    --local_size=<int>      int number of dedalus grid points to include in the mesh passed to spline fit [default: 3]
    
    --precision_phi=<int>   int number of points to sample spline fit at between two grid points along phi [default: 2]
    --precision_r=<int>     int number of points to sample spline fit at between two grid points along r [default: 4]

    --nbinsr=<int>          int number of radial bins to use for radial and 2d distributions [default: 20]
    --nbinsphi=<int>        int number of azimuthal bins to use for 2d distribution [default: 20]
    --t_hist_start=<float>  sim time to start including tracking results in distributions [default: 1.]
    --t_hist_end=<float>    sim time to stop including tracking results in distributions [default: 3.7]
"""
import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)
import dedalus.public as d3
from scipy.interpolate import SmoothSphereBivariateSpline as splinefit
from scipy.stats import norm

### read arguments passed to script ###
print("args read in")
print(args)

file_str = args['<file>'][0]

output_prefix = args['--output']
t_out_start = float(args['--t_out_start'])
t_out_end = float(args['--t_out_end'])

use_cutoff = eval(args['--use_cutoff'])
use_stddev = eval(args['--use_stddev'])
track_all = eval(args['--track_all'])
if eval(args['--r_cutoff']) is None:
    r_cutoff_given = False
else:
    r_cutoff_given = True 

local_size = int(args['--local_size'])
precision_phi = int(args['--precision_phi'])
precision_r = int(args['--precision_r'])
nbinsr = int(args['--nbinsr'])
nbinsphi = int(args['--nbinsphi'])
t_hist_start = float(args['--t_hist_start'])
t_hist_end = float(args['--t_hist_end'])
### read analysis file ###
f = h5py.File(file_str)

### string parsing to identify parameters ###
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

output_suffix = file_str.split('analysis_')[1].split('.')[0].split('/')[0] #[:-1] 
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])
alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
gamma_str = output_suffix.split('gam_')[1].split('_')[0]
eps_str = output_suffix.split('eps_')[1].split('_')[0]
nu_str = output_suffix.split('nu_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
gamma_read = str_to_float(gamma_str)
eps_read = str_to_float(eps_str)
nu_read = str_to_float(nu_str)
alpha_vals = np.array((1e-2, 3.3e-2))
gamma_vals = np.array((0, 30, 85, 240, 400, 675, 1200, 1920, 2500, 3200))
eps_vals = np.array([3.3e-1, 1.0, 2.0])
nu_vals = np.array([5e-5, 2e-4])
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]
gamma = gamma_vals[np.argmin(np.abs(gamma_vals - gamma_read))]
eps = eps_vals[np.argmin(np.abs(eps_vals - eps_read))]
nu = nu_vals[np.argmin(np.abs(nu_vals - nu_read))]    

### define coordinate conversions ###
def r_to_th(r_in, Rfac):
    return np.arcsin(r_in / Rfac)
def th_to_r(th_in, Rfac):
    return Rfac * np.sin(th_in)

### dedalus setup ###
dealias = 3/2
dtype = np.float64
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype = dtype)
disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
edge = disk.edge
radial_basis = disk.radial_basis
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
vort = dist.Field(name = 'vort', bases = disk) # need to specify name for load_from_hdf5 to work below, otherwise need to specify task there

phi_deal_mesh, r_deal_mesh = np.meshgrid(phi_deal[:, 0], r_deal[0, :])
r_mesh = r_deal_mesh.T

Rfactor = 3 # R_planet / R_disk ratio
theta = r_to_th(r_deal[0, :], Rfactor) # well posed only if Rfactor >= 1
phi_mesh, theta_mesh = np.meshgrid(phi_deal[:, 0], theta)
phi_mesh = phi_mesh.T
theta_mesh = theta_mesh.T

#### define all sub-mesh cases (assumes size < floor(N/2)...) ###
#def choose_mesh(lon_mesh, lat_mesh, data, lon_idx, lat_idx, Nlon, Nlat, size, prec, i):
#    bounds = {}
#
#    # lat cut (deal with cases near the pole and near disk edge --- latter should not be an issue on sphere)
#    lat_idx_inner = np.max((0, lat_idx - size))
#    lat_idx_outer = np.min((lat_idx + size, Nlat - 1))
#    lon_sub_mesh_cut1 = lon_mesh[:, lat_idx_inner:lat_idx_outer + 1]
#    lat_sub_mesh_cut1 = lat_mesh[:, lat_idx_inner:lat_idx_outer + 1]
#    data_cut1 = data[:, lat_idx_inner:lat_idx_outer + 1]
#
#    #print(i, lat_idx_inner, lat_idx_outer)
#
#    # remember lat bounds
#    lat_inner = lat_mesh[0, lat_idx_inner]
#    lat_outer = lat_mesh[0, lat_idx_outer]
#    lat_bds = [lat_inner, lat_outer]
#    bounds['lat'] = lat_bds
#    bounds['lat_N'] = (lat_idx_outer - lat_idx_inner) + 1
#    bounds['lat_Nspl'] = int(np.round((bounds['lat_N'] - 1) * (prec + 1) + 1))
#    bounds['lat_endpt'] = True
#
#    if lat_idx_inner == 0:
#        lon_cut = False
#    else:
#        lon_cut = True
#
#    #print(i, lon_cut)
#
#    # lon cut (deal with wrap-around near phi = 0 or 2*pi)
#    if lon_cut and (lon_idx - size < 0): 
#        #print(i, "phi poi is just after 0")
#        lon_idx_wa = 0
#        lon_idx_ea = lon_idx + size
#        lon_idx_wb = Nlon + (lon_idx - size)
#        lon_idx_eb = Nlon - 1
#
#        lon_sub_mesh_cut2a = lon_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
#        lat_sub_mesh_cut2a = lat_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
#        data_cut2a = data_cut1[lon_idx_wa:lon_idx_ea + 1, :]
#
#        lon_sub_mesh_cut2b = lon_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#        lat_sub_mesh_cut2b = lat_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#        data_cut2b = data_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#
#        #lon_sub_mesh = np.vstack((lon_sub_mesh_cut2a, lon_sub_mesh_cut2b))
#        #lat_sub_mesh = np.vstack((lat_sub_mesh_cut2a, lat_sub_mesh_cut2b))
#        #data_sub = np.vstack((data_cut2a, data_cut2b))
#
#        lon_sub_mesh = np.vstack((lon_sub_mesh_cut2b, lon_sub_mesh_cut2a))
#        lat_sub_mesh = np.vstack((lat_sub_mesh_cut2b, lat_sub_mesh_cut2a))
#        data_sub = np.vstack((data_cut2b, data_cut2a))
#
#        # remember lon bounds
#        lon_wa = lon_mesh[lon_idx_wa, 0] # phi = 0
#        lon_ea = lon_mesh[lon_idx_ea, 0]
#        lon_wb = lon_mesh[lon_idx_wb, 0]
#        lon_eb = 2 * np.pi
#        lon_bdsa = [lon_wa, lon_ea]
#        lon_bdsb = [lon_wb, lon_eb]
#        bounds['lon'] = None
#        bounds['lona'] = lon_bdsa
#        bounds['lonb'] = lon_bdsb
#        bounds['lon_N'] = ((lon_idx_ea - lon_idx_wa) + 1) + ((lon_idx_eb - lon_idx_wb) + 1)
#        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
#        bounds['lon_endpt'] = None
#        bounds['lona_N'] = ((lon_idx_ea - lon_idx_wa) + 1)
#        bounds['lonb_N'] = ((lon_idx_eb - lon_idx_wb) + 1)
#        bounds['lona_Nspl'] = int(np.round((lon_idx_ea - lon_idx_wa) * (prec + 1) + 1)) # includes phi = 0
#        bounds['lonb_Nspl'] = int(np.round(((lon_idx_eb - lon_idx_wb) + 1) * (prec + 1))) 
#        bounds['lona_endpt'] = True 
#        bounds['lonb_endpt'] = False # don't include 2*pi as a sample point
#
#    elif lon_cut and (lon_idx + size > Nlon - 1):
#        #print(i, "phi poi is just before 2pi")
#        lon_idx_wa = lon_idx - size
#        lon_idx_ea = Nlon - 1
#        lon_idx_wb = 0
#        lon_idx_eb = (lon_idx + size) - Nlon
#
#        lon_sub_mesh_cut2a = lon_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
#        lat_sub_mesh_cut2a = lat_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
#        data_cut2a = data_cut1[lon_idx_wa:lon_idx_ea + 1, :]  
#
#        lon_sub_mesh_cut2b = lon_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#        lat_sub_mesh_cut2b = lat_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#        data_cut2b = data_cut1[lon_idx_wb:lon_idx_eb + 1, :]
#
#        lon_sub_mesh = np.vstack((lon_sub_mesh_cut2a, lon_sub_mesh_cut2b))
#        lat_sub_mesh = np.vstack((lat_sub_mesh_cut2a, lat_sub_mesh_cut2b))
#        data_sub = np.vstack((data_cut2a, data_cut2b))
#        
#
#        # remember lon bounds
#        lon_wa = lon_mesh[lon_idx_wa, 0]
#        lon_ea = 2 * np.pi
#        lon_wb = lon_mesh[lon_idx_wb, 0] # phi = 0
#        lon_eb = lon_mesh[lon_idx_eb, 0]
#        lon_bdsa = [lon_wa, lon_ea]
#        lon_bdsb = [lon_wb, lon_eb]
#        bounds['lon'] = None
#        bounds['lona'] = lon_bdsa
#        bounds['lonb'] = lon_bdsb
#        bounds['lon_N'] = ((lon_idx_ea - lon_idx_wa) + 1) + ((lon_idx_eb - lon_idx_wb) + 1)
#        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
#        bounds['lon_endpt'] = None
#        bounds['lona_N'] = ((lon_idx_ea - lon_idx_wa) + 1)
#        bounds['lonb_N'] = ((lon_idx_eb - lon_idx_wb) + 1)
#        bounds['lona_Nspl'] = int(np.round(((lon_idx_ea - lon_idx_wa) + 1) * (prec + 1)))
#        bounds['lonb_Nspl'] = int(np.round((lon_idx_eb - lon_idx_wb) * (prec + 1) + 1))
#        bounds['lona_endpt'] = False # don't include 2*pi as a sample point
#        bounds['lonb_endpt'] = True
#
#    elif lon_cut:
#        #print(i, "std phi cut")
#        lon_idx_w = lon_idx - size
#        lon_idx_e = lon_idx + size
#        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
#        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
#        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :]
#
#        # remember lon bounds
#        lon_w = lon_mesh[lon_idx_w, 0]
#        lon_e = lon_mesh[lon_idx_e, 0]
#        lon_bds = [lon_w, lon_e]
#        bounds['lon'] = lon_bds
#        bounds['lona'] = None
#        bounds['lonb'] = None
#        bounds['lon_N'] = (lon_idx_e - lon_idx_w) + 1
#        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
#        bounds['lon_endpt'] = True
#        bounds['lona_N'] = None
#        bounds['lonb_N'] = None
#        bounds['lona_Nspl'] = None
#        bounds['lonb_Nspl'] = None
#        bounds['lona_endpt'] = None
#        bounds['lonb_endpt'] = None
#
#    else: # retain all phi data when near pole (may come back and adjust this choice if too expensive)
#        #print(i, "retain all phi near pole")
#        lon_idx_w = 0
#        lon_idx_e = Nlon - 1
#        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
#        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
#        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :] 
#
#        # remember lon bounds
#        lon_w = lon_mesh[lon_idx_w, 0] 
#        lon_e = 2 * np.pi
#        lon_bds = [lon_w, lon_e]
#        bounds['lon'] = lon_bds
#        bounds['lona'] = None
#        bounds['lonb'] = None
#        bounds['lon_N'] = Nlon
#        bounds['lon_Nspl'] = int(np.round(Nlon * (prec + 1)))
#        bounds['lon_endpt'] = False 
#        bounds['lona_N'] = None
#        bounds['lonb_N'] = None
#        bounds['lona_Nspl'] = None
#        bounds['lonb_Nspl'] = None
#        bounds['lona_endpt'] = None
#        bounds['lonb_endpt'] = None
#
#    return lon_sub_mesh, lat_sub_mesh, data_sub, bounds
#
#def bins_phi(nbins, phis, precision):
#    #phi_w = 0.
#    #phi_e = 2*np.pi
#    #idx_w = 0
#    #idx_e = -1
#    #phi_edges = np.linspace(phi_w, phi_e, nbins + 1, endpoint=True)
#    # shift (rotate) bin edges to not exactly overlap with spline fit test points
#    #phi_edges += (2*np.pi / nbins) * (precision + 1.5)**(-1) 
#    #phi_edges = np.mod(phi_edges, 2 * np.pi)
#    #phi_edges = np.sort(phi_edges)
#    #phi_centers = np.linspace(0.5*(phi_edges[0] + phi_edges[1]), 0.5*(phi_edges[-2] + phi_edges[-1]), nbins, endpoint=True)
#    #phi_centers = np.mod(phi_centers, 2 * np.pi)
#    # return phi_centers, phi_edges
#    rot = (2*np.pi / nbins) * (precision + 1.5)**(-1)
#    phi_wa = 0. + rot
#    phi_ea = 2 * np.pi
#    phi_wb = 0.
#    phi_eb = 0. + rot
#    phi_edges_a = np.linspace(phi_wa, phi_ea - (2*np.pi / nbins) * (1 - (precision + 1.5)**(-1)), nbins, endpoint=True) #np.linspace(phi_wa, phi_ea, nbins + 1, endpoint=True)
#    phi_edges_a = np.concatenate((phi_edges_a, [phi_ea]))
#    phi_edges_b = np.array((phi_wb, phi_eb))
#    phi_centers = np.linspace(0.5*(phi_edges_a[0] + phi_edges_a[1]), 0.5*(phi_edges_a[-2] + (2 * np.pi + phi_edges_b[1])), nbins, endpoint=True)
#    dphis = np.concatenate((np.diff(phi_edges_a[:-1]), [(2 * np.pi + phi_edges_b[1]) - phi_edges_a[-2]])) # correspnd to the angular width of each bin, indexed as phi_centers #is 
#    print(phi_wa, phi_ea, phi_wb, phi_eb, phi_edges_a, phi_edges_b, phi_centers, dphis)
#    return phi_centers, phi_edges_a, phi_edges_b, dphis

#def bins_r(nbins, rs, r_outer):    
#    r_inner = 0.
#    idx_inner = np.where(rs >= r_inner)[0][0]
#    idx_outer = np.where(rs <= r_outer)[0][-1]
#    r_edges = np.linspace(r_inner, r_outer, nbins + 1, endpoint=True)
#    r_centers = np.linspace(0.5*(r_edges[0] + r_edges[1]), 0.5*(r_edges[-2] + r_edges[-1]), nbins, endpoint=True)
#    return r_centers, r_edges

### determine subset of grid points and vorticity data to pass to spline fit ###
def choose_mesh(lon_mesh, lat_mesh, data, lon_idx, lat_idx, Nlon, Nlat, size):

    ### store additional info for later calls to test and bins functions ###
    bounds = {}

    ### lat cut
    lat_idx_inner = np.max((0, lat_idx - size))
    lat_idx_outer = np.min((lat_idx + size, Nlat - 1))
    lon_sub_mesh_cut1 = lon_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    lat_sub_mesh_cut1 = lat_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    data_cut1 = data[:, lat_idx_inner:lat_idx_outer + 1]

    bounds['lat_idxs'] = [lat_idx_inner, lat_idx_outer]
    bounds['lat_sub_mesh_g'] = lat_sub_mesh_cut1[0, :]
    bounds['lat_pole_flag'] = lat_idx - size < 0

    # if close to pole, retain all points in phi
    if lat_idx_inner == 0:
        lon_cut = False
    else:
        lon_cut = True
    
    ### lon cut
    if lon_cut and (lon_idx - size < 0): 
        lon_idx_wa = 0
        lon_idx_ea = lon_idx + size
        lon_idx_wb = Nlon + (lon_idx - size)
        lon_idx_eb = Nlon - 1

        lon_sub_mesh_cut2a = lon_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
        lat_sub_mesh_cut2a = lat_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
        data_cut2a = data_cut1[lon_idx_wa:lon_idx_ea + 1, :]

        lon_sub_mesh_cut2b = lon_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
        lat_sub_mesh_cut2b = lat_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
        data_cut2b = data_cut1[lon_idx_wb:lon_idx_eb + 1, :]

        lon_sub_mesh = np.vstack((lon_sub_mesh_cut2b, lon_sub_mesh_cut2a))
        lat_sub_mesh = np.vstack((lat_sub_mesh_cut2b, lat_sub_mesh_cut2a))
        data_sub = np.vstack((data_cut2b, data_cut2a))

        bounds['lon_idxs'] = None
        bounds['lon_std_flag'] = None
        bounds['lon_sub_mesh_g'] = lon_sub_mesh[:, 0]
        bounds['lon_a_idxs'] = [lon_idx_wa, lon_idx_ea]
        bounds['lon_b_idxs'] = [lon_idx_wb, lon_idx_eb]
        bounds['lon_ab_flag'] = True # True to include a endpt and exclude b endpt, False for vice versa
        bounds['lon_a_bds'] = [lon_sub_mesh_cut2a[0, 0], lon_sub_mesh_cut2a[-1, 0]]
        bounds['lon_b_bds'] = [lon_sub_mesh_cut2b[0, 0], 2 * np.pi]

    elif lon_cut and (lon_idx + size > Nlon - 1):
        lon_idx_wa = lon_idx - size
        lon_idx_ea = Nlon - 1
        lon_idx_wb = 0
        lon_idx_eb = (lon_idx + size) - Nlon

        lon_sub_mesh_cut2a = lon_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
        lat_sub_mesh_cut2a = lat_sub_mesh_cut1[lon_idx_wa:lon_idx_ea + 1, :]
        data_cut2a = data_cut1[lon_idx_wa:lon_idx_ea + 1, :]  

        lon_sub_mesh_cut2b = lon_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
        lat_sub_mesh_cut2b = lat_sub_mesh_cut1[lon_idx_wb:lon_idx_eb + 1, :]
        data_cut2b = data_cut1[lon_idx_wb:lon_idx_eb + 1, :]

        lon_sub_mesh = np.vstack((lon_sub_mesh_cut2a, lon_sub_mesh_cut2b))
        lat_sub_mesh = np.vstack((lat_sub_mesh_cut2a, lat_sub_mesh_cut2b))
        data_sub = np.vstack((data_cut2a, data_cut2b))
       
        bounds['lon_idxs'] = None
        bounds['lon_std_flag'] = None
        bounds['lon_bds'] = None
        bounds['lon_a_idxs'] = [lon_idx_wa, lon_idx_ea]
        bounds['lon_b_idxs'] = [lon_idx_wb, lon_idx_eb]
        bounds['lon_ab_flag'] = False # True to include a endpt and exclude b endpt, False for vice versa
        bounds['lon_a_bds'] = [lon_sub_mesh_cut2a[0, 0], 2 * np.pi]
        bounds['lon_b_bds'] = [lon_sub_mesh_cut2b[0, 0], lon_sub_mesh_cut2b[-1, 0]]

    elif lon_cut:
        lon_idx_w = lon_idx - size
        lon_idx_e = lon_idx + size
        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :]

        bounds['lon_idxs'] = [lon_idx_w, lon_idx_e]
        bounds['lon_std_flag'] = True # whether to include endpt in test_pts
        bounds['lon_bds'] = [lon_sub_mesh[0, 0], lon_sub_mesh[-1, 0]]
        bounds['lon_a_idxs'] = None
        bounds['lon_b_idxs'] = None
        bounds['lon_ab_flag'] = None
        bounds['lon_a_bds'] = None
        bounds['lon_b_bds'] = None

    else: # retain all phi data when near pole (may come back and adjust this choice if too expensive)
        lon_idx_w = 0
        lon_idx_e = Nlon - 1
        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :] 

        bounds['lon_idxs'] = [lon_idx_w, lon_idx_e]
        bounds['lon_std_flag'] = False # whether to include endpt in test_pts
        bounds['lon_bds'] = [0, 2 * np.pi]
        bounds['lon_a_idxs'] = None
        bounds['lon_b_idxs'] = None
        bounds['lon_ab_flag'] = None
        bounds['lon_a_bds'] = None
        bounds['lon_b_bds'] = None

    return lon_sub_mesh, lat_sub_mesh, data_sub, bounds

def lat_test(lat_sub_mesh_g, lat_idxs, prec, pole_flag):
    lat_idx_inner, lat_idx_outer = lat_idxs 
    if pole_flag:
        test_pts = np.linspace(0, lat_sub_mesh_g[0], prec + 1, endpoint=False)[1:] # remove r = 0 here, will add it as a test point later
    else:
        test_pts = np.array([])
    for i in range(lat_idx_outer - lat_idx_inner):
        test_pts = np.concatenate((test_pts, np.linspace(lat_sub_mesh_g[i], lat_sub_mesh_g[i + 1], prec + 1, endpoint=False)))
    test_pts = np.concatenate((test_pts, [lat_sub_mesh_g[lat_idx_outer - lat_idx_inner]]))
    return test_pts

def lon_test_std(lon_bds, lon_idxs, prec, std_flag):
    lon_idx_w, lon_idx_e = lon_idxs
    endpt = std_flag
    N = np.round((prec + 1) * (lon_idx_e - lon_idx_w + int(not std_flag))) + int(std_flag)
    test_pts = np.linspace(lon_bds[0], lon_bds[-1], N, endpoint=endpt)
    return test_pts

def lon_test_ab(lon_a_bds, lon_b_bds, lon_a_idxs, lon_b_idxs, prec, ab_flag):
    lon_idx_wa, lon_idx_ea = lon_a_idxs
    lon_idx_wb, lon_idx_eb = lon_b_idxs
    endpt_a = ab_flag
    endpt_b = not ab_flag
    N_a = np.round((prec + 1) * (lon_idx_ea - lon_idx_wa + int(not ab_flag))) + int(ab_flag)
    N_b = np.round((prec + 1) * (lon_idx_eb - lon_idx_wb + int(ab_flag))) + int(not ab_flag)
    test_pts_a = np.linspace(lon_a_bds[0], lon_a_bds[-1], N_a, endpoint=endpt_a)
    test_pts_b = np.linspace(lon_b_bds[0], lon_b_bds[-1], N_b, endpoint=endpt_b)
    return test_pts_a, test_pts_b

def bins_r(r_g, prec, r_idx_outer):
    center_pts = np.linspace(0, r_g[0], prec + 1, endpoint=False) 
    for i in range(1, r_idx_outer + 1):
        center_pts = np.concatenate((center_pts, np.linspace(r_g[i - 1], r_g[i], prec + 1, endpoint=False)))
    center_pts = np.concatenate((center_pts, [r_g[r_idx_outer]]))
    #drs = np.diff(center_pts)
    edge_pts = [0]
    for ii in range(1, center_pts.shape[0]):
        edge_pts.append(0.5*(center_pts[ii - 1] + center_pts[ii]))
    #edge_pts.append(0.5*(center_pts[-2] + center_pts[-1]) + drs[-1])
    edge_pts.append(0.5*(center_pts[-2] + center_pts[-1]) + (center_pts[-1] - center_pts[-2]))
    edge_pts = np.array(edge_pts)
    #edge_pts = np.concatenate((edge_pts, np.linspace(0.5*(center_pts[0] + center_pts[1]), 0.5*(center_pts[-2] + center_pts[-1]) + drs[-1], center_pts.shape[0], endpoint=True)))
    #drs = np.concatenate((drs, [drs[-1]]))
    drs = np.diff(edge_pts)
    return center_pts, edge_pts, drs

def bins_phi(phi_g, prec):
    center_pts = np.linspace(0, 2*np.pi, phi_g.shape[0] * (prec + 1), endpoint=False)
    dphis = np.diff(center_pts)
    edge_pts_main = np.linspace(0.5*(center_pts[0] + center_pts[1]), 2*np.pi - 0.5*dphis[-1], center_pts.shape[0], endpoint=True)
    edge_pts_0a = np.array([edge_pts_main[-1], 2*np.pi]) # together with edge_pts_0b, define the edges of 1 bin (2pi and 0 are not actually edges)
    edge_pts_0b = np.array([0, edge_pts_main[0]])
    dphis = np.concatenate((dphis, [dphis[-1]]))
    return center_pts, edge_pts_main, edge_pts_0a, edge_pts_0b, dphis


### specify writes to process ###
t = f['tasks/KE'].dims[0]['sim_time'][:]
ws = np.arange(np.where(t <= t_out_start)[0][-1], np.where(t >= t_out_end)[0][0] + 1)
nw = len(ws) # number of writes to process
tw = t[ws]

# determine Lgamma if needed
if use_cutoff and (not r_cutoff_given):
    tdur = 30 #0.2 #30 - a damping time - would be ideal
    tend = t[-1] # by default will look closest to the latest times available
    startidx = np.where(t >= tend - tdur)[0][0]
    endidx = -1 #np.where(t >= tend)[0][0]
    EN = np.array(f['tasks/EN'])
    EN_tavg = np.mean(EN[startidx:endidx])
    KE_tavg = ((eps/np.pi) - nu*EN_tavg) / (2*alpha) # analytical estimate, if tracking before sim is converged
    u_rms = np.sqrt(2 * KE_tavg)
    L_gamma = (u_rms / gamma)**(1/3)
    r_cutoff = 2 * L_gamma #0.6 #2 * L_gamma #0.6 #np.min((2 * L_gamma, 0.6)) # we can come back and modify this choice if we get some better wisdom on this / review jet landscape
elif use_cutoff and r_cutoff_given:
    r_cutoff = float(args['--r_cutoff'])

print("r_cutoff", r_cutoff)
if use_cutoff:
    cutoff_mask = r_mesh >= r_cutoff
    r_cutoff_idx = np.where(r_deal[0, :] < r_cutoff)[0][-1] # idx of last r to keep

### begin processing ###
vort_mus = []
vort_stddevs = []

lat_poi_idxs = []
lon_poi_idxs = []
lat_pois = []
lon_pois = []

vort_maxs = []
th_locs = []
r_locs = []
phi_locs = []

grid = 0
grid_is = []
grid_rlocs = []
grid_rpoi_idxs = []
non_grid_is = []
non_grid_rlocs = []
non_grid_rpoi_idxs = []

prog_cad = 32
for i, w in enumerate(ws):
    if i % prog_cad == 0:
        print('writes loop: i = %d out of %d' %(i, nw))

    # load vorticity grid data
    vort.load_from_hdf5(f, w)
    vort_g = np.copy(vort['g'])
    
    # fit grid data to normal distribution
    mu_fit, stddev_fit = norm.fit(vort_g)
    vort_mus.append(mu_fit)
    vort_stddevs.append(stddev_fit)
    stddev_mask = np.abs(vort_g) <= 2*stddev_fit

    # apply filters
    if use_cutoff:
        vort_g[cutoff_mask] = 0.
    
    # fit grid data to normal distribution
    #mu_fit, stddev_fit = norm.fit(vort_g[vort_g != 0.])
    #vort_mus.append(mu_fit)
    #vort_stddevs.append(stddev_fit)
    #stddev_mask = np.abs(vort_g) <= 2*stddev_fit
    
    if use_stddev:
        vort_g[stddev_mask] = 0.

    # identify point(s) of interest -- plan to revisit this to look for all pockets of vorticity, not just the most prominent one
    lon_poi_idx, lat_poi_idx = np.where(vort_g == np.max(vort_g))[0][0], np.where(vort_g == np.max(vort_g))[1][0]
    
    # if lat poi coincides with r_cutoff, attempt to refine the cutoff region
    #print(i, use_cutoff, lat_poi_idx, r_cutoff_idx)
    if use_cutoff and lat_poi_idx == r_cutoff_idx:
        refine = True
        vort_g_refine = np.copy(vort['g'])
        r_cutoff_refine = r_cutoff
        while refine:
            r_cutoff_refine *= 0.9
            r_cutoff_refine_idx = np.where(r_deal[0, :] < r_cutoff_refine)[0][-1]
            print(i, "refining", "r_cutoff_refine =", r_cutoff_refine)
            cutoff_mask_refine = r_mesh >= r_cutoff_refine
            vort_g_refine[cutoff_mask_refine] = 0.
            lon_poi_idx_refine, lat_poi_idx_refine = np.where(vort_g_refine == np.max(vort_g_refine))[0][0], np.where(vort_g_refine == np.max(vort_g_refine))[1][0]
            if lat_poi_idx_refine != r_cutoff_refine_idx:
                refine = False
                lon_poi_idx = lon_poi_idx_refine
                lat_poi_idx = lat_poi_idx_refine
                vort_g = vort_g_refine
                print("exiting refine successfully")
            elif lat_poi_idx_refine == r_cutoff_refine_idx and r_cutoff_refine < 0.1:
                refine = False
                print("exiting refine unsuccessfully")
                raise

    lon_poi_idxs.append(lon_poi_idx)
    lat_poi_idxs.append(lat_poi_idx)
    lon_pois.append(phi_mesh[lon_poi_idx, 0])
    lat_pois.append(theta_mesh[0, lat_poi_idx])

    # determine local mesh to pass to spline fit
    Nphi_deal = int(np.round(dealias * Nphi))
    Nr_deal = int(np.round(dealias * Nr))
    #lon_sub_mesh, lat_sub_mesh, data_sub, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g, lon_poi_idx, lat_poi_idx, Nphi_deal, Nr_deal, local_size, precision)
    #lon_sub_mesh, lat_sub_mesh, data_sub, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g, lon_poi_idx, lat_poi_idx, Nphi_deal, Nr_deal, local_size, precision, i)
    lon_sub_mesh, lat_sub_mesh, data_sub, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g, lon_poi_idx, lat_poi_idx, Nphi_deal, Nr_deal, local_size)

    ## order should not matter... but let's see what happens
    #lon_sub_mesh = lon_sub_mesh.T
    #lat_sub_mesh = lat_sub_mesh.T
    #data_sub = data_sub.T

    if i == 3040:
        print("phi_mesh", phi_mesh.shape, phi_mesh)
        print("theta_mesh", theta_mesh.shape, theta_mesh)
        
        print("lon_sub_mesh", lon_sub_mesh)
        print("lat_sub_mesh", lat_sub_mesh)
        print("data_sub", data_sub)

    # call spline fit
    #if i == 2:
    #    print( lon_sub_mesh, lat_sub_mesh, data_sub, bounds)


    lats_spl = lat_sub_mesh.ravel()
    lons_spl = lon_sub_mesh.ravel()
    data_in = data_sub.ravel()
    #weights = stddev_fit**(-1) * np.ones(len(data_in))
    mu_in, stddev_in = norm.fit(data_in)
    weights = stddev_in**(-1) * np.ones(len(data_in))
    #s = len(data_in) - np.sqrt(2*len(data_in))
    s = 7
    if i == 3040:
        print("lats_spl", lats_spl.shape, lats_spl)
        print("lons_spl", lons_spl.shape, lons_spl)
        print("data_in", data_in.shape, data_in)
        print("weights", weights.shape, weights)
        print("s", s)
    try:
        spl_out = splinefit(lats_spl, lons_spl, data_in, w=weights, s=s)
    except:
        try:
            s = 10
            spl_out = splinefit(lats_spl, lons_spl, data_in, w=weights, s=s)
        except:
            try:
                s = 15
                spl_out = splinefit(lats_spl, lons_spl, data_in, w=weights, s=s)
            except:
                s = 49
                spl_out = splinefit(lats_spl, lons_spl, data_in, w=weights, s=s)


    #spl_out = splinefit(lats_spl, lons_spl, data_in, s=0)

    # sampling points to desired precision
    #lats_test = np.linspace(bounds['lat'][0], bounds['lat'][1], bounds['lat_Nspl'], endpoint = bounds['lat_endpt'])    
    #if bounds['lon'] is None: # area includes the 2*pi to 0 crossing 
    #    lonsa_test = np.linspace(bounds['lona'][0], bounds['lona'][1], bounds['lona_Nspl'], endpoint = bounds['lona_endpt'])
    #    lonsb_test = np.linspace(bounds['lonb'][0], bounds['lonb'][1], bounds['lonb_Nspl'], endpoint = bounds['lonb_endpt'])
    #    data_test_a = spl_out(lats_test, lonsa_test)
    #    data_test_b = spl_out(lats_test, lonsb_test)
    #    if bounds['lonb_endpt']:
    #        #lons_test = np.concatenate((lonsa_test, lonsb_test))
    #        data_test = np.hstack((data_test_a, data_test_b))
    #    else:
    #        #lons_test = np.concatenate((lonsb_test, lonsa_test))
    #        data_test = np.hstack((data_test_b, data_test_a))
    #else:
    #    lons_test = np.linspace(bounds['lon'][0], bounds['lon'][1], bounds['lon_Nspl'], endpoint = bounds['lon_endpt'])
    #    data_test = spl_out(lats_test, lons_test)
    
    #try:
    #    data_test = spl_out(lats_test, lons_test)
    #except:
    #    print(lats_test, lons_test)
    #    raise

    lats_test = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], precision_r, bounds['lat_pole_flag'])
    if i == 3040:
        print("lats_test", lats_test)
    if bounds['lon_idxs'] is None:
        lons_a_test, lons_b_test = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'], bounds['lon_a_idxs'], bounds['lon_b_idxs'], precision_phi, bounds['lon_ab_flag'])
        #Lons_a_test, Lats_a_test = np.meshgrid(lons_a_test, lats_test)
        #Lons_b_test, Lats_b_test = np.meshgrid(lons_b_test, lats_test)
        #Lons_a_test = Lons_a_test.T
        #Lats_a_test = Lats_a_test.T
        #Lons_b_test = Lons_b_test.T
        #Lats_b_test = Lats_b_test.T
        data_test_a = spl_out(lats_test, lons_a_test)
        data_test_b = spl_out(lats_test, lons_b_test)
        #data_test_a = spl_out(Lats_a_test.ravel(), Lons_a_test.ravel())
        #data_test_b = spl_out(Lats_b_test.ravel(), Lons_b_test.ravel())
        if bounds['lon_ab_flag']:
            data_test = np.hstack((data_test_b, data_test_a))
            if i == 3040:
                print("b->a")
                print("lons_a_test", lons_a_test)
                print("lons_b_test", lons_b_test)
                print("data_test_a", data_test_a)
                print("data_test_b", data_test_b)
                print("data_test", data_test)
        else:
            data_test = np.hstack((data_test_a, data_test_b))
            if i == 3040:
                print("a->b")
                print("lons_a_test", lons_a_test)
                print("lons_b_test", lons_b_test)
                print("data_test_a", data_test_a)
                print("data_test_b", data_test_b)
                print("data_test", data_test)
    else:
        lons_test = lon_test_std(bounds['lon_bds'], bounds['lon_idxs'], precision_phi, bounds['lon_std_flag'])
        #Lons_test, Lats_test = np.meshgrid(lons_test, lats_test)
        #Lons_test = Lons_test.T
        #Lats_test = Lats_test.T
        data_test = spl_out(lats_test, lons_test)
        #data_test = spl_out(Lats_test.ravel(), Lons_test.ravel())
        if i == 3040:
            print("std")
            print("lons_test", lons_test)
            print("data_test", data_test)
    #if bounds['lat_pole_flag']:
        #data_test = np.concatenate((data_test, [spl_out(0, 0)]))
        #lats_test = np.concatenate((lats_test, [0])) 
        #if i == 3040:
        #    print("lat_pole_flag")
        #    print("data_test", data_test)
        #    print("lats_test", lats_test)
        #data_test_pole = spl_out(0, 0)

    if i == 3040:
        print("shape of data_test", data_test.shape)

    # find new max and keep information
    data_max = np.max(data_test)

    lat_max_idx = np.where(data_test == data_max)[0][0]
    lat_loc = lats_test[lat_max_idx]
    if bounds['lon_idxs'] is None:
        if data_max in data_test_a:
            lon_max_idx_a = np.where(data_test_a == data_max)[1][0]
            lon_loc = lons_a_test[lon_max_idx_a]
        elif data_max in data_test_b:
            lon_max_idx_b = np.where(data_test_b == data_max)[1][0]
            lon_loc = lons_b_test[lon_max_idx_b]
        else:
            print("This should never happen")
            raise
    else:
        lon_max_idx = np.where(data_test == data_max)[1][0]
        lon_loc = lons_test[lon_max_idx]
    r_loc = th_to_r(lat_loc, Rfactor)

    if bounds['lat_pole_flag']:
        data_test_pole = spl_out(0, 0)
        if data_test_pole > data_max:
            lat_loc = 0
            r_loc = 0
            lon_loc = None # hopefully we can deal with this correctly in the 2d hist case...  
    vort_maxs.append(data_max)
    th_locs.append(lat_loc)
    r_locs.append(r_loc)
    phi_locs.append(lon_loc)

    #if r_loc in r_deal:
    match = False
    for rr in r_deal[0, :]:
        if np.isclose(r_loc, rr):
            match = True
    if match:
        grid += 1
        grid_is.append(i)
        grid_rlocs.append(r_loc)
        grid_rpoi_idxs.append(lat_poi_idx)
    else: #it appears that these are only not "in" because they slightly differ due to floating pt errors
        non_grid_is.append(i)
        non_grid_rlocs.append(r_loc)
        non_grid_rpoi_idxs.append(lat_poi_idx)

print("grid", grid)
for j in range(len(grid_is)):
    print(grid_is[j], grid_rlocs[j], r_deal[0, grid_rpoi_idxs[j]], np.isclose(grid_rlocs[j], r_deal[0, grid_rpoi_idxs[j]]))
print("non_grid")
for j in range(len(non_grid_is)):
    print(non_grid_is[j], non_grid_rlocs[j], r_deal[0, non_grid_rpoi_idxs[j]], np.isclose(non_grid_rlocs[j], r_deal[0, non_grid_rpoi_idxs[j]]))
print("grid pts")
for j in range(r_deal[0, :].shape[0]):
    print(j, r_deal[0, j])


#print(data_test)

### time-averaged distribution ###
#if use_cutoff:
#    r_outer = r_cutoff
#else:
#    r_outer = 1.
###phi_centers, phi_edges = bins_phi(nbinsphi, phi_deal, precision)
#phi_centers, phi_edges_a, phi_edges_b, dphis = bins_phi(nbinsphi, phi_deal, precision)
#r_centers, r_edges = bins_r(nbinsr, r_deal, r_outer)
#r_inner = r_edges[:-1]
#r_outer = r_edges[1:]
###phi_w = phi_edges[:-1]
###phi_e = phi_edges[1:]

r_centers, r_edges, drs = bins_r(r_deal[0, :], precision_r, r_cutoff_idx)
phi_centers, phi_edges_main, phi_edges_0a, phi_edges_0b, dphis = bins_phi(phi_deal[:, 0], precision_phi)

r_inner = r_edges[:-1]
r_outer = r_edges[1:]
#print(r_inner, r_outer)
phi_w_main = phi_edges_main[:-1]
phi_e_main = phi_edges_main[1:]
phi_w_0a, phi_e_0a = phi_edges_0a
phi_w_0b, phi_e_0b = phi_edges_0b

bin_area_fracs_r = r_outer**2 - r_inner**2 # area of shell / area of unit disk # we could normalize based on the area contained within r_cutoff instead
bin_area_fracs_phi = dphis / (2 * np.pi) # when multiplied by bin_area_frac_r gives: area of 2d bin / area of unit disk

phi_centers_2d, r_centers_2d = np.meshgrid(phi_centers, r_centers)
phi_centers_2d = phi_centers_2d.T

hist_r = np.zeros_like(r_centers)
hist_phi = np.zeros_like(phi_centers)
hist_2d = np.zeros_like(phi_centers_2d)
n_hist = 0

hist_r_rep = np.zeros_like(r_centers)
hist_phi_rep = np.zeros_like(phi_centers)
hist_2d_rep = np.zeros_like(phi_centers_2d)

bin_area_fracs_2d = np.zeros_like(phi_centers_2d) # area of bin / area of unit disk # we could normalize based on the area contained within r_cutoff instead
for ii in range(bin_area_fracs_2d.shape[0]):
    for jj in range(bin_area_fracs_2d.shape[1]):
        bin_area_fracs_2d[ii, jj] = bin_area_fracs_r[jj] * bin_area_fracs_phi[ii]

### I would like to ensure the area fracs and the way the points are being defined and such are all what we expect - ToDo


###

ws_hist = np.arange(np.where(t <= t_hist_start)[0][-1], np.where(t >= t_hist_end)[0][0] + 1)

## temporary for debugging
i_list = []
r_mask_sums = []
phi_mask_sums = []
##


grid_in_hist = 0
non_grid_in_hist = 0
for j, w_hist in enumerate(ws_hist):
    if w_hist in ws:

        i = np.where(ws == w_hist)[0][0]
        r_loc_i = r_locs[i]
        phi_loc_i = phi_locs[i]
        
        if i in grid_is:
            grid_in_hist += 1
        if i in non_grid_is:
            non_grid_in_hist += 1

        i_list.append(i)
        r_mask = np.logical_and(r_loc_i > r_inner, r_loc_i < r_outer)
        phi_mask_main = np.logical_and(phi_loc_i > phi_w_main, phi_loc_i < phi_e_main)
        phi_mask_0a = phi_loc_i > phi_w_0a and phi_loc_i < phi_e_0a
        phi_mask_0b = phi_loc_i >= phi_w_0b and phi_loc_i < phi_e_0b
        phi_mask_0 = np.array([phi_mask_0a or phi_mask_0b])
        phi_mask = np.concatenate((phi_mask_0, phi_mask_main))

        ##
        if np.sum(r_mask) != 1 or np.sum(phi_mask) != 1:
            np.set_printoptions(precision=15)
            print(i)
            print(r_loc_i, r_inner, r_outer)
            print(r_mask)
            print(np.sum(r_mask))
            print(np.array(phi_loc_i), phi_w, phi_e)
            print(phi_mask)
            print(np.sum(phi_mask))
            print("")
        r_mask_sums.append(np.sum(r_mask))
        phi_mask_sums.append(np.sum(phi_mask))
        ##

        hist_r[r_mask] += 1
        hist_phi[phi_mask] += 1
        hist_2d[phi_mask, r_mask] += 1
        n_hist += 1

        hist_r_rep[r_mask] += 1 #bin_area_fracs_r[r_mask]  #1/bin_area_fracs_r[r_mask]
        hist_phi_rep[phi_mask] += 1 #bin_area_fracs_phi[phi_mask] #1/bin_area_fracs_phi[phi_mask]
        hist_2d_rep[phi_mask, r_mask] += 1 #bin_area_fracs_2d[phi_mask, r_mask] #1/bin_area_fracs_2d[phi_mask, r_mask]

hist_r_rep *= 1/n_hist
hist_phi_rep *= 1/n_hist
hist_2d_rep *= 1/n_hist

#print(len(i_list), np.unique(i_list).shape)
#print(np.sum(np.array(r_mask_sums) != 1))
#print(np.sum(np.array(phi_mask_sums) != 1))

print(grid_in_hist, non_grid_in_hist)
#for j in range(r_centers.shape[0]):
#    print(j, r_centers[j], r_edges[j], r_edges[j+1], drs[j])

### output ###
processed = {}

processed['nw'] = nw
processed['ws'] = ws
processed['tw'] = tw

processed['r_deal'] = r_deal
processed['phi_deal'] = phi_deal

processed['vort_mus'] = vort_mus
processed['vort_stddevs'] = vort_stddevs

processed['lat_poi_idxs'] = lat_poi_idxs
processed['lon_poi_idxs'] = lon_poi_idxs
processed['lat_pois'] = lat_pois
processed['lon_pois'] = lon_pois

processed['vort_maxs'] = vort_maxs
processed['th_locs'] = th_locs
processed['r_locs'] = r_locs
processed['phi_locs'] = phi_locs

processed['ws_hist'] = ws_hist
processed['r_centers'] = r_centers
processed['r_edges'] = r_edges
processed['drs'] = drs
processed['phi_centers'] = phi_centers
processed['phi_edges_main'] = phi_edges_main
processed['phi_edges_0a'] = phi_edges_0a
processed['phi_edges_0b'] = phi_edges_0b
processed['dphis'] = dphis
processed['r_centers_2d'] = r_centers_2d
processed['phi_centers_2d'] = phi_centers_2d
processed['hist_r'] = hist_r
processed['hist_phi'] = hist_phi
processed['hist_2d'] = hist_2d
processed['n_hist'] = n_hist

processed['hist_r_rep'] = hist_r_rep
processed['hist_phi_rep'] = hist_phi_rep
processed['hist_2d_rep'] = hist_2d_rep

print('saving processed results as: ' + output_prefix + '_' + output_suffix + '.npy')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)





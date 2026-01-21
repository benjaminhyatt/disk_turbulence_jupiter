"""
Usage:
    process_tracking.py <file>... [options]

Options:    
    --output=<str>          prefix in the name of the output file [default: processed_tracking]

    --t_out_start=<float>   sim time to begin tracking [default: 0.3]
    --t_out_end=<float>     sim time to stop tracking [default: 3.7]

    --use_cutoff=<bool>     flag True to ignore grid data greater than a specified radius [default: True]
    --use_stddev=<bool>     flag True to ignore grid data of size less than a multiple of the standard deviation [default: True]
    --track_all=<bool>      flag True to track all vorticity extrema (recommend using both options above) [default: False]
    --r_cutoff=<float>      provide a specified cutoff radius, if None, default will base cutoff on Lgamma [default: None]

    --local_size=<int>      int number of dedalus grid points to include in the mesh passed to spline fit [default: 3]
    --precision=<int>       int number of points to sample spline fit at between two grid points along one coord [default: 4]

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
precision = int(args['--precision'])

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

### define all sub-mesh cases (assumes size < floor(N/2)...) ###
def choose_mesh(lon_mesh, lat_mesh, data, lon_idx, lat_idx, Nlon, Nlat, size, prec):
    bounds = {}

    # lat cut (deal with cases near the pole and near disk edge --- latter should not be an issue on sphere)
    lat_idx_inner = np.max((0, lat_idx - size))
    lat_idx_outer = np.min((lat_idx + size, Nlat - 1))
    lon_sub_mesh_cut1 = lon_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    lat_sub_mesh_cut1 = lat_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    data_cut1 = data[:, lat_idx_inner:lat_idx_outer + 1]

    # remember lat bounds
    lat_inner = lat_mesh[0, lat_idx_inner]
    lat_outer = lat_mesh[0, lat_idx_outer]
    lat_bds = [lat_inner, lat_outer]
    bounds['lat'] = lat_bds
    bounds['lat_N'] = (lat_idx_outer - lat_idx_inner) + 1
    bounds['lat_Nspl'] = int(np.round((bounds['lat_N'] - 1) * (prec + 1) + 1))
    bounds['lat_endpt'] = True

    if lat_idx_inner == 0:
        lon_cut = False
    else:
        lon_cut = True

    # lon cut (deal with wrap-around near phi = 0 or 2*pi)
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

        #lon_sub_mesh = np.vstack((lon_sub_mesh_cut2a, lon_sub_mesh_cut2b))
        #lat_sub_mesh = np.vstack((lat_sub_mesh_cut2a, lat_sub_mesh_cut2b))
        #data_sub = np.vstack((data_cut2a, data_cut2b))

        lon_sub_mesh = np.vstack((lon_sub_mesh_cut2b, lon_sub_mesh_cut2a))
        lat_sub_mesh = np.vstack((lat_sub_mesh_cut2b, lat_sub_mesh_cut2a))
        data_sub = np.vstack((data_cut2b, data_cut2a))

        # remember lon bounds
        lon_wa = lon_mesh[lon_idx_wa, 0] # phi = 0
        lon_ea = lon_mesh[lon_idx_ea, 0]
        lon_wb = lon_mesh[lon_idx_wb, 0]
        lon_eb = 2 * np.pi
        lon_bdsa = [lon_wa, lon_ea]
        lon_bdsb = [lon_wb, lon_eb]
        bounds['lon'] = None
        bounds['lona'] = lon_bdsa
        bounds['lonb'] = lon_bdsb
        bounds['lon_N'] = ((lon_idx_ea - lon_idx_wa) + 1) + ((lon_idx_eb - lon_idx_wb) + 1)
        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
        bounds['lon_endpt'] = None
        bounds['lona_N'] = ((lon_idx_ea - lon_idx_wa) + 1)
        bounds['lonb_N'] = ((lon_idx_eb - lon_idx_wb) + 1)
        bounds['lona_Nspl'] = int(np.round((lon_idx_ea - lon_idx_wa) * (prec + 1) + 1)) # includes phi = 0
        bounds['lonb_Nspl'] = int(np.round(((lon_idx_eb - lon_idx_wb) + 1) * (prec + 1))) 
        bounds['lona_endpt'] = True 
        bounds['lonb_endpt'] = False # don't include 2*pi as a sample point

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
        

        # remember lon bounds
        lon_wa = lon_mesh[lon_idx_wa, 0]
        lon_ea = 2 * np.pi
        lon_wb = lon_mesh[lon_idx_wb, 0] # phi = 0
        lon_eb = lon_mesh[lon_idx_eb, 0]
        lon_bdsa = [lon_wa, lon_ea]
        lon_bdsb = [lon_wb, lon_eb]
        bounds['lon'] = None
        bounds['lona'] = lon_bdsa
        bounds['lonb'] = lon_bdsb
        bounds['lon_N'] = ((lon_idx_ea - lon_idx_wa) + 1) + ((lon_idx_eb - lon_idx_wb) + 1)
        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
        bounds['lon_endpt'] = None
        bounds['lona_N'] = ((lon_idx_ea - lon_idx_wa) + 1)
        bounds['lonb_N'] = ((lon_idx_eb - lon_idx_wb) + 1)
        bounds['lona_Nspl'] = int(np.round(((lon_idx_ea - lon_idx_wa) + 1) * (prec + 1)))
        bounds['lonb_Nspl'] = int(np.round((lon_idx_eb - lon_idx_wb) * (prec + 1) + 1))
        bounds['lona_endpt'] = False # don't include 2*pi as a sample point
        bounds['lonb_endpt'] = True

    elif lon_cut:
        lon_idx_w = lon_idx - size
        lon_idx_e = lon_idx + size
        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :]

        # remember lon bounds
        lon_w = lon_mesh[lon_idx_w, 0]
        lon_e = lon_mesh[lon_idx_e, 0]
        lon_bds = [lon_w, lon_e]
        bounds['lon'] = lon_bds
        bounds['lona'] = None
        bounds['lonb'] = None
        bounds['lon_N'] = (lon_idx_e - lon_idx_w) + 1
        bounds['lon_Nspl'] = int(np.round((bounds['lon_N'] - 1) * (prec + 1) + 1))
        bounds['lon_endpt'] = True
        bounds['lona_N'] = None
        bounds['lonb_N'] = None
        bounds['lona_Nspl'] = None
        bounds['lonb_Nspl'] = None
        bounds['lona_endpt'] = None
        bounds['lonb_endpt'] = None

    else: # retain all phi data when near pole (may come back and adjust this choice if too expensive)
        lon_idx_w = 0
        lon_idx_e = Nlon - 1
        lon_sub_mesh = lon_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        lat_sub_mesh = lat_sub_mesh_cut1[lon_idx_w:lon_idx_e + 1, :]
        data_sub = data_cut1[lon_idx_w:lon_idx_e + 1, :] 

        # remember lon bounds
        lon_w = lon_mesh[lon_idx_w, 0] 
        lon_e = 2 * np.pi
        lon_bds = [lon_w, lon_e]
        bounds['lon'] = lon_bds
        bounds['lona'] = None
        bounds['lonb'] = None
        bounds['lon_N'] = Nlon
        bounds['lon_Nspl'] = int(np.round(Nlon * (prec + 1)))
        bounds['lon_endpt'] = False 
        bounds['lona_N'] = None
        bounds['lonb_N'] = None
        bounds['lona_Nspl'] = None
        bounds['lonb_Nspl'] = None
        bounds['lona_endpt'] = None
        bounds['lonb_endpt'] = None

    return lon_sub_mesh, lat_sub_mesh, data_sub, bounds

def bins_phi(nbins, phis):
    phi_w = 0.
    phi_e = 2*np.pi
    idx_w = 0
    idx_e = -1
    phi_edges = np.linspace(phi_w, phi_e, nbins + 1, endpoint=True)
    phi_centers = np.linspace(0.5*(phi_edges[0] + phi_edges[1]), 0.5*(phi_edges[-2] + phi_edges[-1]), nbins, endpoint=True)
    return phi_centers, phi_edges

def bins_r(nbins, rs, r_outer):    
    r_inner = 0.
    idx_inner = np.where(rs >= r_inner)[0][0]
    idx_outer = np.where(rs <= r_outer)[0][-1]
    r_edges = np.linspace(r_inner, r_outer, nbins + 1, endpoint=True)
    r_centers = np.linspace(0.5*(r_edges[0] + r_edges[1]), 0.5*(r_edges[-2] + r_edges[-1]), nbins, endpoint=True)
    return r_centers, r_edges

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
    if use_stddev:
        vort_g[stddev_mask] = 0.

    # identify point(s) of interest -- plan to revisit this to look for all pockets of vorticity, not just the most prominent one
    lon_poi_idx, lat_poi_idx = np.where(vort_g == np.max(vort_g))[0][0], np.where(vort_g == np.max(vort_g))[1][0]
    lon_poi_idxs.append(lon_poi_idx)
    lat_poi_idxs.append(lat_poi_idx)
    lon_pois.append(phi_mesh[lon_poi_idx, 0])
    lat_pois.append(theta_mesh[0, lat_poi_idx])

    # determine local mesh to pass to spline fit
    Nphi_deal = int(np.round(dealias * Nphi))
    Nr_deal = int(np.round(dealias * Nr))
    lon_sub_mesh, lat_sub_mesh, data_sub, bounds = choose_mesh(phi_mesh, theta_mesh, vort_g, lon_poi_idx, lat_poi_idx, Nphi_deal, Nr_deal, local_size, precision)
    
    # call spline fit
    lats_spl = lat_sub_mesh.ravel()
    lons_spl = lon_sub_mesh.ravel()
    data_in = data_sub.ravel()
    weights = stddev_fit**(-1) * np.ones(len(data_in))
    s = len(data_in)

    try:
        spl_out = splinefit(lats_spl, lons_spl, data_in, w=weights, s=s)
    except:
        print("given to choose_mesh")
        print(phi_mesh, theta_mesh, vort_g, lon_poi_idx, lat_poi_idx, Nphi_deal, Nr_deal, local_size, precision)
        print("given to splinefit")
        print(lats_spl)
        print(lons_spl)
        print(data_in)
        print(weights)
        print(s)
        raise

    # sampling points to desired precision
    lats_test = np.linspace(bounds['lat'][0], bounds['lat'][1], bounds['lat_Nspl'], endpoint = bounds['lat_endpt'])
    if bounds['lon'] is None: # area includes the 2*pi to 0 crossing 
        lonsa_test = np.linspace(bounds['lona'][0], bounds['lona'][1], bounds['lona_Nspl'], endpoint = bounds['lona_endpt'])
        lonsb_test = np.linspace(bounds['lonb'][0], bounds['lonb'][1], bounds['lonb_Nspl'], endpoint = bounds['lonb_endpt'])
        data_test_a = spl_out(lats_test, lonsa_test)
        data_test_b = spl_out(lats_test, lonsb_test)
        if bounds['lonb_endpt']:
            #lons_test = np.concatenate((lonsa_test, lonsb_test))
            data_test = np.hstack((data_test_a, data_test_b))
        else:
            #lons_test = np.concatenate((lonsb_test, lonsa_test))
            data_test = np.hstack((data_test_b, data_test_a))
    else:
        lons_test = np.linspace(bounds['lon'][0], bounds['lon'][1], bounds['lon_Nspl'], endpoint = bounds['lon_endpt'])
        data_test = spl_out(lats_test, lons_test)
    
    #try:
    #    data_test = spl_out(lats_test, lons_test)
    #except:
    #    print(lats_test, lons_test)
    #    raise

    # find new max and keep information
    data_max = np.max(data_test)
    lat_max_idx = np.where(data_test == data_max)[0][0]
    lon_max_idx = np.where(data_test == data_max)[1][0]
    lat_loc = lats_test[lat_max_idx]
    lon_loc = lons_test[lon_max_idx]
    r_loc = th_to_r(lat_loc, Rfactor)

    vort_maxs.append(data_max)
    th_locs.append(lat_loc)
    r_locs.append(r_loc)
    phi_locs.append(lon_loc)

### time-averaged distribution ###
if use_cutoff:
    r_outer = r_cutoff
else:
    r_outer = 1.
phi_centers, phi_edges = bins_phi(nbinsphi, phi_deal)
r_centers, r_edges = bins_r(nbinsr, r_deal, r_outer)
r_inner = r_edges[:-1]
r_outer = r_edges[1:]
phi_w = phi_edges[:-1]
phi_e = phi_edges[1:]

bin_area_fracs_r = r_outer**2 - r_inner**2

phi_centers_2d, r_centers_2d = np.meshgrid(phi_centers, r_centers)
phi_centers_2d = phi_centers_2d.T

hist_r = np.zeros_like(r_centers)
hist_2d = np.zeros_like(phi_centers_2d)
n_hist = 0

hist_r_rep = np.zeros_like(r_centers)
hist_2d_rep = np.zeros_like(phi_centers_2d)

bin_area_fracs_2d = np.zeros_like(phi_centers_2d)
for ii in range(bin_area_fracs_2d.shape[0]):
    for jj in range(bin_area_fracs_2d.shape[1]):
        bin_area_fracs_2d[ii, jj] = bin_area_fracs_r[jj] * 0.5 * (phi_e[ii] - phi_w[ii])

ws_hist = np.arange(np.where(t <= t_hist_start)[0][-1], np.where(t >= t_hist_end)[0][0] + 1)
for j, w_hist in enumerate(ws_hist):
    if w_hist in ws:
        i = np.where(ws == w_hist)[0][0]

        r_loc_i = r_locs[i]
        phi_loc_i = phi_locs[i]

        r_mask = np.logical_and(r_loc_i >= r_inner, r_loc_i <= r_outer)
        phi_mask = np.logical_and(phi_loc_i >= phi_w, phi_loc_i <= phi_e)

        hist_r[r_mask] += 1
        hist_2d[phi_mask, r_mask] += 1
        n_hist += 1

        hist_r_rep[r_mask] += bin_area_fracs_r[r_mask]
        hist_2d_rep[phi_mask, r_mask] += bin_area_fracs_2d[phi_mask, r_mask]

hist_r_rep *= 1/n_hist
hist_2d_rep *= 1/n_hist

### output ###
processed = {}

processed['nw'] = nw
processed['ws'] = ws
processed['tw'] = tw

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
processed['phi_centers'] = phi_centers
processed['phi_edges'] = phi_edges
processed['r_centers_2d'] = r_centers_2d
processed['phi_centers_2d'] = phi_centers_2d
processed['hist_r'] = hist_r
processed['hist_2d'] = hist_2d
processed['n_hist'] = n_hist

processed['hist_r_rep'] = hist_r_rep
processed['hist_2d_rep'] = hist_2d_rep

print('saving processed results as: ' + output_prefix + '_' + output_suffix + '.npy')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)





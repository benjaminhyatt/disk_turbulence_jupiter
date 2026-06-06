import numpy as np
import dedalus.public as d3
from scipy.interpolate import RectSphereBivariateSpline as splinefit
from scipy.stats import norm
from scipy.optimize import minimize

Rfactor = 3

def r_to_th(r_in, Rfac):
    return np.arcsin(r_in / Rfac)
def th_to_r(th_in, Rfac):
    return Rfac * np.sin(th_in)

### determine subset of grid points and vorticity data to pass to spline fit ###
def choose_mesh(lon_mesh, lat_mesh, data, lon_idx, lat_idx, Nlon, Nlat, size_lon, size_lat):

    ### store additional info for later calls to test and bins functions ###
    bounds = {}

    ### lat cut
    lat_idx_inner = np.max((0, lat_idx - size_lat))
    lat_idx_outer = np.min((lat_idx + size_lat, Nlat - 1))
    lon_sub_mesh_cut1 = lon_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    lat_sub_mesh_cut1 = lat_mesh[:, lat_idx_inner:lat_idx_outer + 1]
    data_cut1 = data[:, lat_idx_inner:lat_idx_outer + 1]

    bounds['lat_idxs'] = [lat_idx_inner, lat_idx_outer]
    bounds['lat_sub_mesh_g'] = lat_sub_mesh_cut1[0, :]
    bounds['lat_pole_flag'] = lat_idx - size_lat < 0

    # if close to pole, retain all points in phi
    if lat_idx_inner == 0:
        lon_cut = False
    else:
        lon_cut = True
    
    ### lon cut
    if lon_cut and (lon_idx - size_lon < 0): 
        lon_idx_wa = 0
        lon_idx_ea = lon_idx + size_lon
        lon_idx_wb = Nlon + (lon_idx - size_lon)
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

    elif lon_cut and (lon_idx + size_lon > Nlon - 1):
        lon_idx_wa = lon_idx - size_lon
        lon_idx_ea = Nlon - 1
        lon_idx_wb = 0
        lon_idx_eb = (lon_idx + size_lon) - Nlon

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
        lon_idx_w = lon_idx - size_lon
        lon_idx_e = lon_idx + size_lon
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

def lat_test(lat_sub_mesh_g, lat_idxs, prec, include_near_pole):
    lat_idx_inner, lat_idx_outer = lat_idxs 
    if include_near_pole:
        test_pts = np.linspace(0, lat_sub_mesh_g[0], prec + 1, endpoint=False)[1:] # r=0 itself will be included in the test set later
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
    
def find_max_opt(spl, bounds, prec_r, prec_phi):
    
    spl_func = lambda x: -1 * spl(x[0], x[1], dtheta=0, dphi=0)
    spl_jac = lambda x:  (-1 * spl(x[0], x[1], dtheta=1, dphi=0), np.sin(x[0])**(-1) * -1 * spl(x[0], x[1], dtheta=0, dphi=1)) # ensure x[0] (theta) is not at pole
    
    # loop through local grid points as initial guesses
    lat_guesses_1 = []
    lon_guesses_1 = []
    lat_results_1 = []
    lon_results_1 = []
    data_results_1 = []

    #temporary
    opt_bds_save = []

    lats_guess = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], 0, bounds['lat_pole_flag'])
    if bounds['lon_idxs'] is None: # search region can be restricted in phi to an area close to the 2pi to 0 transition
        lons_a_guess, lons_b_guess = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'], bounds['lon_a_idxs'], bounds['lon_b_idxs'], 0, bounds['lon_ab_flag'])
        lons_a_guess[lons_a_guess >= np.pi] = lons_a_guess[lons_a_guess >= np.pi] - 2 * np.pi
        lons_b_guess[lons_b_guess >= np.pi] = lons_b_guess[lons_b_guess >= np.pi] - 2 * np.pi
        lons_a_guess_resort = np.argsort(lons_a_guess)
        lons_b_guess_resort = np.argsort(lons_b_guess)
        lons_a_guess = lons_a_guess[lons_a_guess_resort]
        lons_b_guess = lons_b_guess[lons_b_guess_resort]
        for lat in lats_guess:
            for lon in lons_a_guess:
                x0 = (lat, lon)
                if bounds['lon_ab_flag']:
                    opt_bds = ((lats_guess[0], lats_guess[-1]), (lons_b_guess[0], lons_a_guess[-1]))
                else:
                    opt_bds = ((lats_guess[0], lats_guess[-1]), (lons_a_guess[0], lons_b_guess[-1]))
                opt_bds_save.append(opt_bds)
                opt_result = minimize(spl_func, x0, bounds=opt_bds, jac=spl_jac, method='L-BFGS-B', tol=1e-3)
                lat_guesses_1.append(lat)
                lon_guesses_1.append(lon)
                lat_results_1.append(opt_result.x[0])
                lon_results_1.append(opt_result.x[1])
                data_results_1.append(-1 * spl_func(opt_result.x))
            for lon in lons_b_guess:
                x0 = (lat, lon)
                if bounds['lon_ab_flag']:
                    opt_bds = ((lats_guess[0], lats_guess[-1]), (lons_b_guess[0], lons_a_guess[-1]))
                else:
                    opt_bds = ((lats_guess[0], lats_guess[-1]), (lons_a_guess[0], lons_b_guess[-1]))
                opt_bds_save.append(opt_bds)
                opt_result = minimize(spl_func, x0, bounds=opt_bds, jac=spl_jac, method='L-BFGS-B', tol=1e-3)
                lat_guesses_1.append(lat)
                lon_guesses_1.append(lon)
                lat_results_1.append(opt_result.x[0])
                lon_results_1.append(opt_result.x[1])
                data_results_1.append(-1 * spl_func(opt_result.x))
    else:
        lons_guess = lon_test_std(bounds['lon_bds'], bounds['lon_idxs'], 0, bounds['lon_std_flag']) 
        lons_guess[lons_guess >= np.pi] = lons_guess[lons_guess >= np.pi] - 2 * np.pi
        lons_guess_resort = np.argsort(lons_guess)
        lons_guess = lons_guess[lons_guess_resort]  
        for lat in lats_guess:
            for lon in lons_guess:
                x0 = (lat, lon)
                opt_bds = ((lats_guess[0], lats_guess[-1]), (lons_guess[0], lons_guess[-1]))
                opt_bds_save.append(opt_bds)
                opt_result = minimize(spl_func, x0, bounds=opt_bds, jac=spl_jac, method='L-BFGS-B', tol=1e-3)
                lat_guesses_1.append(lat)
                lon_guesses_1.append(lon)
                lat_results_1.append(opt_result.x[0])
                lon_results_1.append(opt_result.x[1])
                data_results_1.append(-1 * spl_func(opt_result.x))
        
    data_max = np.max(data_results_1)
    max_idx = np.where(data_results_1 == data_max)[0][0]
    lat_loc = lat_results_1[max_idx]
    r_loc = th_to_r(lat_loc, Rfactor)
    lon_loc = lon_results_1[max_idx]
    if lon_loc < 0 and lon_loc >= -np.pi:
        lon_loc += 2 * np.pi
    
    return data_max, lat_loc, r_loc, lon_loc

def find_max_sample(spl, bounds, prec_r, prec_phi):
    lats_test = lat_test(bounds['lat_sub_mesh_g'], bounds['lat_idxs'], precision_r, bounds['lat_pole_flag'])
    if bounds['lon_idxs'] is None:
        lons_a_test, lons_b_test = lon_test_ab(bounds['lon_a_bds'], bounds['lon_b_bds'], bounds['lon_a_idxs'], bounds['lon_b_idxs'], precision_phi, bounds['lon_ab_flag'])
        lons_a_test[lons_a_test >= np.pi] = lons_a_test[lons_a_test >= np.pi] - 2 * np.pi
        lons_b_test[lons_b_test >= np.pi] = lons_b_test[lons_b_test >= np.pi] - 2 * np.pi
        lons_a_test_resort = np.argsort(lons_a_test)
        lons_b_test_resort = np.argsort(lons_b_test)
        lons_a_test = lons_a_test[lons_a_test_resort]
        lons_b_test = lons_b_test[lons_b_test_resort]
        data_test_a = spl(lats_test, lons_a_test)
        data_test_b = spl(lats_test, lons_b_test)
        if bounds['lon_ab_flag']:
            data_test = np.hstack((data_test_b, data_test_a))
        else:
            data_test = np.hstack((data_test_a, data_test_b))
    else:
        lons_test = lon_test_std(bounds['lon_bds'], bounds['lon_idxs'], precision_phi, bounds['lon_std_flag'])
        lons_test[lons_test >= np.pi] = lons_test[lons_test >= np.pi] - 2 * np.pi
        lons_test_resort = np.argsort(lons_test)
        lons_test = lons_test[lons_test_resort]
        data_test = spl(lats_test, lons_test)

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
        data_test_pole = spl(0, 0)
        if data_test_pole > data_max:
            lat_loc = 0
            r_loc = 0
            lon_loc = rand.uniform(0, 2*np.pi) # choice is arbitrary for hist_r, but does affect hist_phi and hist_2d...
    if lon_loc < 0 and lon_loc >= -np.pi:
        lon_loc += 2 * np.pi

    return data_max, lat_loc, r_loc, lon_loc

def bins_r(r_g, prec, width, r_idx_outer):
    n_g = r_idx_outer + 1
    r_g_aug = np.concatenate(([0], r_g))
        
    n_g_aug = n_g + 1
    test_pts_global = np.array([])
    for i in range(n_g_aug - 1):
        test_pts_global = np.concatenate((test_pts_global, np.linspace(r_g_aug[i], r_g_aug[i + 1], prec + 1, endpoint=False)))
    test_pts_global = np.concatenate((test_pts_global, [r_g_aug[n_g_aug - 1]]))
    if test_pts_global.shape[0] != n_g * (prec + 1) + 1:
        print("This should never happen")
        raise

    n_edges = int(np.ceil((n_g_aug) / width)) + 1
    bin_edges = [0]
    for i in range(1, n_edges - 1):
        bin_edges.append(0.5 * (test_pts_global[int(width * (i - 1) * (prec + 1)) + int(prec/2)] + test_pts_global[int(width * (i - 1) * (prec + 1)) + int(prec/2) + 1]))
    bin_edges.append(r_g_aug[n_g_aug - 1])
    bin_edges = np.array(bin_edges)

    n_test_per_bin = []
    for i in range(n_edges - 1):
        n_test_per_bin.append(np.sum(np.logical_and(test_pts_global > bin_edges[i], test_pts_global < bin_edges[i + 1])))
    n_test_per_bin = np.array(n_test_per_bin)
    if np.unique(n_test_per_bin).shape[0] > 1:
        print("The options specified (e.g., precisions and bin widths) resulted in a non-uniform number of test points per bin.")
        print("The results will be re-weighted accordingly.")
    
    bin_centers = []
    for i in range(n_edges - 1):
        bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    bin_centers = np.array(bin_centers)

    drs = np.diff(bin_edges)

    test_pts_global_edges = [0]
    for ii in range(1, int((n_g_aug - 1)*(prec + 1)) + 1):
        test_pts_global_edges.append(0.5*(test_pts_global[ii - 1] + test_pts_global[ii]))
    test_pts_global_edges.append(0.5*(test_pts_global[-2] + test_pts_global[-1]) + (test_pts_global[-1] - test_pts_global[-2]))
    test_pts_global_edges = np.array(test_pts_global_edges)

    return bin_centers, bin_edges, drs, n_test_per_bin, test_pts_global, test_pts_global_edges

def bins_phi(phi_g, prec, width):
    n_g = phi_g.shape[0]
    test_pts_global = np.linspace(0, 2*np.pi, n_g * (prec + 1), endpoint=False)
    
    n_edges = int(np.ceil(n_g / width))
    bin_edges_main = []
    for i in range(1, n_edges):
        bin_edges_main.append(0.5*(phi_g[int(width * (i - 1))] + phi_g[int(width * (i - 1)) + 1]))
    if int(width * (n_edges - 1) + 1) <= n_g - 1:
        bin_edges_main.append(0.5*(phi_g[int(width * (n_edges - 1))] + phi_g[int(width * (n_edges - 1)) + 1]))
    else:
        bin_edges_main.append(0.5*(phi_g[int(width * (n_edges - 1))] + 2 * np.pi))
    bin_edges_main = np.array(bin_edges_main)

    bin_edges_0a = np.array([bin_edges_main[-1], 2 * np.pi])
    bin_edges_0b = np.array([0, bin_edges_main[0]])

    n_test_per_bin = []
    for i in range(n_edges - 1):
        n_test_per_bin.append(np.sum(np.logical_and(test_pts_global > bin_edges_main[i], test_pts_global < bin_edges_main[i + 1])))
    
    n_test_0a = np.sum(np.logical_and(test_pts_global > bin_edges_0a[0], test_pts_global <= bin_edges_0a[1]))
    n_test_0b = np.sum(np.logical_and(test_pts_global >= bin_edges_0b[0], test_pts_global < bin_edges_0b[1]))
    n_test_per_bin.append(n_test_0a + n_test_0b)
    n_test_per_bin = np.array(n_test_per_bin)
    if np.unique(n_test_per_bin).shape[0] > 1:
        print("The options specified (e.g., precisions and bin widths) resulted in a non-uniform number of test points per bin.")
        print("The results will be re-weighted accordingly.")

    bin_centers = []
    for i in range(n_edges - 1):
        bin_centers.append(0.5 * (bin_edges_main[i] + bin_edges_main[i + 1]))
    bin_centers.append(np.mod(0.5 * (bin_edges_main[-1] + (2 * np.pi + bin_edges_main[0])), 2 * np.pi))
    bin_centers = np.array(bin_centers)

    dphis = np.concatenate((np.diff(bin_edges_main), [(2 * np.pi + bin_edges_main[0]) - bin_edges_main[-1]]))

    test_pts_global_edges_main = np.linspace(0.5*(test_pts_global[0] + test_pts_global[1]), 0.5*(test_pts_global[-1] + 2*np.pi), n_g * (prec + 1), endpoint=True)
    test_pts_global_edges_0a = np.array([test_pts_global_edges_main[-1], 2*np.pi])
    test_pts_global_edges_0b = np.array([0., test_pts_global_edges_main[0]])
    return bin_centers, bin_edges_main, bin_edges_0a, bin_edges_0b, dphis, n_test_per_bin, test_pts_global, test_pts_global_edges_main, test_pts_global_edges_0a, test_pts_global_edges_0b
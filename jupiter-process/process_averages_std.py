import numpy as np
import dedalus.public as d3
#import matplotlib.pyplot as plt
import h5py
from dedalus.extras import plot_tools
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def gaussian(x, a, mu, sig):
    return a * np.exp(-0.5*((x-mu)/sig)**2) / np.sqrt(2*np.pi*(sig**2))

def gaussian_origin(x, a, sig):
    return gaussian(x, a, 0., sig)

# Read in parameters
Nphi, Nr = 512, 256
nu = 2e-4
#gamma = 85
k_force = 20
alpha = 1e-2
amp = 1
ring = 0
restart_evolved = False
eps = amp**2

gammas = [85, 240, 400, 675, 1920]

processed = {}
for gamma in gammas: 
    print(gamma)
    processed[gamma] = {}

    output_suffix = 'nu_{:.0e}'.format(nu) + '_gam_{:.1e}'.format(gamma) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr) 
    output_suffix += '_eps_{:.1e}'.format(eps)
    output_suffix += '_alpha_{:.1e}'.format(alpha)
    output_suffix += '_ring_{:d}'.format(ring)
    output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
    output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

    f = h5py.File('../jupiter-run/analysis_' + output_suffix + '/analysis_' + output_suffix + '_s1.h5')
    t = np.array(f['tasks/u'].dims[0]['sim_time'])

    dealias = 3/2
    dtype = np.float64
    coords = d3.PolarCoordinates('phi', 'r')
    dist = d3.Distributor(coords, dtype = dtype)
    disk = d3.DiskBasis(coords, shape = (Nphi, Nr), radius = 1, dealias = dealias, dtype = dtype)
    edge = disk.edge
    radial_basis = disk.radial_basis
    phi, r = dist.local_grids(disk)
    phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))
    vort = dist.Field(name = 'vort', bases = disk)
    vort_pile = dist.Field(name = 'vort_pile', bases = disk)
    radial_vort_pile = dist.Field(name = 'radial_vort_pile', bases = radial_basis)

    # briefly get L_gam
    tdur = 10
    tendidx = -1
    tend = t[tendidx]
    tstartidx = np.where(t >= tend - tdur)[0][0]
    tstart = t[tstartidx]
    #KE = f['tasks/KE'][:, 0, 0]
    EN = f['tasks/EN'][:, 0, 0]
    #KE_tavg = np.mean(KE[tstartidx:tendidx])
    EN_tavg = np.mean(EN[tstartidx:tendidx])
    u_rms = ((eps/np.pi) - (nu*EN_tavg))/(2 * alpha)
    L_gam = (u_rms/gamma)**(1/3)
    processed[gamma]['L_gam'] = L_gam
    #print(gamma, L_gam, 3*L_gam)
    # end of getting L_gam


    
    t2 = t[-1]
    t1 = t2 - 2e2
    print(gamma, t1, t2)
    tidxs = np.where(np.logical_and(t >= t1, t <= t2))[0]
    nidxs = tidxs.shape[0]
    prog_cad = 25
    print(nidxs, tidxs)
    vort_pile.change_scales(dealias)

    Phi, R = plot_tools.quad_mesh(phi_deal[:, 0], r_deal[0, :])
    X = (R * np.cos(Phi)).T
    Y = (R * np.sin(Phi)).T

    phinodes, rnodes = np.meshgrid(phi_deal[:, 0], r_deal[0, :])
    rnodes = rnodes.T
    for idx in tidxs:
        if (idx % prog_cad) == 0:
            print(idx)
        vort.load_from_hdf5(f, idx)
        vort_max = vort['g'].max()
        vort_pile['g'] += (1/nidxs) * np.copy(vort['g'])

    norm_vort_pile = d3.integ(vort_pile*vort_pile).evaluate()['g'][0, 0]
    vort_pile['g'] /= norm_vort_pile

    Z = np.copy(vort_pile['g'])

    processed[gamma]['X'] = X
    processed[gamma]['Y'] = Y
    processed[gamma]['Z'] = Z
    processed[gamma]['t'] = t

    vort_pile_m0 = d3.Average(vort_pile, coords['phi']).evaluate()['g']
    radial_norm = simpson(vort_pile_m0*vort_pile_m0, r_deal[0, :]) # treating like an ordinary variable, not weighted in r
    vort_pile_m0 /= radial_norm
    Z_m0 = np.copy(vort_pile_m0)
    #if gamma == 1920 or gamma == 400:
    #    Z_m0_opt, Z_m0_cov = curve_fit(gaussian_origin, np.concatenate((-1 * r_deal[0, -1::-1], r_deal[0, :])), np.concatenate((vort_pile_m0[0, -1::-1], vort_pile_m0[0, :])), p0 = (1, 0.25)) # silly but correct
    #    Z_m0_amp, Z_m0_var = Z_m0_opt
    #    Z_m0_expect = 0. # we are stipulating this
    #else:
    #    Z_m0_opt, Z_m0_cov = curve_fit(gaussian, r_deal[0, :], vort_pile_m0[0, :], p0 = (1, 0.5, 0.25))
    #    Z_m0_amp, Z_m0_expect, Z_m0_var = Z_m0_opt

    processed[gamma]['r'] = r
    processed[gamma]['r_deal'] = r_deal
    processed[gamma]['Z_m0'] = Z_m0
    #processed[gamma]['Z_m0_amp'] = Z_m0_amp
    #processed[gamma]['Z_m0_expect'] = Z_m0_expect
    #processed[gamma]['Z_m0_var'] = Z_m0_var

#fig, ax = plt.subplots(figsize=(7, 6))
#lim = max(abs(Z.min()), abs(Z.max()))
#mesh = ax.pcolormesh(X, Y, Z, shading='auto', cmap='RdBu_r', vmin = -lim, vmax = lim)
#fig.colorbar(mesh, ax=ax)
#ax.plot(0.439*np.cos(np.linspace(0,2*np.pi)), 0.439*np.sin(np.linspace(0, 2*np.pi)))

#plt.savefig('average_' + output_suffix + '.png')

output_suffix = 'nu_{:.0e}'.format(nu) + '_kf_{:.1e}'.format(k_force) + '_Nphi_{:}'.format(Nphi) + '_Nr_{:}'.format(Nr)
output_suffix += '_eps_{:.1e}'.format(eps)
output_suffix += '_alpha_{:.1e}'.format(alpha)
output_suffix += '_ring_{:d}'.format(ring)
output_suffix += '_restart_evolved_{:d}'.format(restart_evolved)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

np.save('processed_averages_std_' + output_suffix + '.npy', processed)

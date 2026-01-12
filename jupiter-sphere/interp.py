import h5py
import matplotlib.pyplot as plt
import numpy as np
import dedalus.public as d3
from scipy.interpolate import RectBivariateSpline

# Forcing parameters
epsilon = 1     # Energy injection rate
kf = 256        # Forcing wavenumber
kfw = 2         # Forcing bandwidth
seed = None     # Random seed

# Derived parameters
eta = epsilon * kf**2  # Enstrophy injection rate

Nphi = 2048
Ntheta = 1024
dealias = 3/2
dtype = np.float64
R = 1

epsilon = 0.237
gamma = 2*1186
alpha = 1/30
U = np.sqrt(epsilon/alpha)
L_g = (U/gamma)**(1/3)


start, end = 2, 1108
savename_func = lambda write: 'write_{:06}.png'.format(write)
# loading vorticity, stream function psi
fig, ax = plt.subplots()
#fig2, ax2 = plt.subplots()
t = np.zeros(end-start+1)
#sd = np.zeros(end-start+1)
nd = []
for i in range(start, end+1):
    with h5py.File(f'snapshots/snapshots_s{i}.h5', mode='r') as snapshots:
        time = snapshots['tasks']['psi'].dims[0]['sim_time'][0]
        t[i-start] = time
        dset = snapshots['tasks']['vorticity']
        w = dset[0] #shape: (2048, 1024)
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        theta = np.flip(theta)
        #"unwrap local"
        phi_diff = np.diff(phi)
        theta_diff = np.diff(theta)
        phi_jumps = np.where(np.abs(phi_diff) > np.pi)
        theta_jumps = np.where(np.abs(theta_diff) > np.pi/2)
        for j in phi_jumps[0]: 
            phi[j+1:] += np.sign(phi_diff[j])*2*np.pi
        for j in theta_jumps[0]:
            theta[j+1:] += np.sign(theta_diff[j])*2*np.pi
        gidx = np.argmin(np.abs(L_g-theta)) + 20
        if gidx > theta.size//2:
            gidx = theta.size-gidx
        #print(gidx, gidx2)
        #print("L_g distance from pole")
        #print(gidx, theta[gidx])
        npcap = w[:, -gidx:]
        #print(spcap.shape, npcap.shape)
        maxdx = np.unravel_index(npcap.argmax(), npcap.shape) #positive vorticity
        #x = np.sort(np.array([maxdx[0]-2, maxdx[0]-1, maxdx[0], maxdx[0]+1, maxdx[0]+2])%int(len(phi)))
        #y = np.sort(np.array([maxdx[1]-2, maxdx[1]-1, maxdx[1], maxdx[1]+1, maxdx[1]+2])%int(len(theta)))
        #z = np.zeros((5, 5))
        #print(x, y)
        #for i in range(5):
        #    for j in range(5):
        #        z[i, j] = w[int(x[i]), int(y[j])]
        #print(z) #array of vorticities
        x_idx = np.arange(maxdx[0]-2, maxdx[0]+3)%int(len(phi))
        y_idx = np.arange(maxdx[1]-2, maxdx[1]+3)%int(len(theta))
        
        x = phi[x_idx]
        y = theta[y_idx]
        print('before unwrap')
        print(x, y)
        #"unwrap local"
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        x_jumps = np.where(np.abs(x_diff) > np.pi)
        y_jumps = np.where(np.abs(y_diff) > np.pi/2)
        for j in x_jumps[0]:
            x[j+1:] += -1*np.sign(x_diff[j])*2*np.pi
        for j in y_jumps[0]:
            y[j+1:] += -1*np.sign(y_diff[j])*np.pi
        print('after unwrap')
        print(x, y)
        z = w[np.ix_(x_idx, y_idx)]

        spline = RectBivariateSpline(x, y, z, kx=3, ky=3)
        
        x_new = np.linspace(x[2]-0.2, x[2]+0.2, 10)
        y_new = np.linspace(y[2]-0.2, y[2]+0.2, 10)
        
        new_data = spline(x_new, y_new)
        print('spline new data below: ')
        print(new_data[:10, :10])
        cont_max, cont_argmax = np.max(new_data), np.argmax(new_data)
        row, col = np.unravel_index(cont_argmax, new_data.shape)
        #print(mindx[1], maxdx[1])
        #ndist = np.pi - theta[-gidx:][int(maxdx[1])]
        print(f'new_data shape: {new_data.shape}')
        #want to find distance from pole corresponding to max vorticity
        print(cont_argmax)
        pos = [x_new[row], y_new[col]]
        print(f'position pos: {pos}')
        ndist = np.sqrt(pos[0]**2 + pos[1]**2)
        print(f'max dist from spline: {ndist}')
        nd.append(ndist)
        #print(f'before save: i = {i}')
nd = np.array(nd)
np.save('northpole.npy', nd)

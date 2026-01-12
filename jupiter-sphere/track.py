import h5py
import matplotlib.pyplot as plt
import numpy as np
import dedalus.public as d3

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


start, end = 200, 1108
savename_func = lambda write: 'write_{:06}.png'.format(write)
# loading vorticity, stream function psi
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
t = np.zeros(end-start+1)
sd = np.zeros(end-start+1)
nd = np.zeros(end-start+1)
for i in range(start, end+1):
    with h5py.File(f'snapshots/snapshots_s{i}.h5', mode='r') as snapshots:
        time = snapshots['tasks']['psi'].dims[0]['sim_time'][0]
        t[i-start] = time
        dset = snapshots['tasks']['vorticity']
        w = dset[0] #shape: (2048, 1024)
        phi = dset.dims[1][0][:].ravel()
        theta = dset.dims[2][0][:].ravel()
        theta = np.flip(theta)
        gidx = np.argmin(np.abs(L_g-theta)) + 20
        if gidx > theta.size//2:
            gidx = theta.size-gidx
        #print(gidx, gidx2)
        #print("L_g distance from pole")
        #print(gidx, theta[gidx])
        spcap = w[:, :gidx]
        npcap = w[:, -gidx:]
        #print(spcap.shape, npcap.shape)
        mindx = np.unravel_index(spcap.argmin(), spcap.shape) #negative vorticity
        maxdx = np.unravel_index(npcap.argmax(), npcap.shape) #positive vorticity
        print(maxdx)
        print(mindx[1], maxdx[1])
        sdist = theta[:gidx][int(mindx[1])]
        ndist = np.pi - theta[-gidx:][int(maxdx[1])]
        #print(mindx, maxdx)
        sd[i-start] = sdist
        nd[i-start] = ndist
        print(f'before save: i = {i}')
np.save('northpole.py', nd)
np.save('southpole.py', sd)
#ax.plot(t, nd, color='r', label='N')
#ax.plot(t, sd, color='b', label='S')
ax.hist(nd, bins=100)
ax2.hist(sd, bins=100)
ax.set_title('central vortex distance from pole over time')
ax.set_xlabel('distance')
ax.set_ylabel('frequency')
ax.legend()
ax2.set_title('central vortex distance from pole over time')
ax2.set_xlabel('distance')
ax2.set_ylabel('frequency')
ax2.legend()
#fig.savefig('vtrack.png', dpi=100)
fig.savefig('np_hist.png', dpi=100)
fig2.savefig('sp_hist.png', dpi=100)
plt.close(fig)

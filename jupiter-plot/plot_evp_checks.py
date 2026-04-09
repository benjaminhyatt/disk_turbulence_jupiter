"""
Usage:
    plot_evp_checks.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the file being loaded [default: processed_rossby_evp]
"""


import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
import pathlib

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

print("args read in")
print(args)

file_str = args['<file>'][0]
output_prefix = args['--output']
output_suffix = file_str.split(output_prefix + '_')[1].split('.')[0].split('/')[0] #[:-1] 
#print(output_suffix)

alpha_str = output_suffix.split('alpha_')[1].split('_')[0]
alpha_read = str_to_float(alpha_str)
alpha_vals = np.array((1e-2, 3.3e-2, 1e-1))
alpha = alpha_vals[np.argmin(np.abs(alpha_vals - alpha_read))]

kf_str = output_suffix.split('kf_')[1].split('_')[0]
kf_read = str_to_float(kf_str)
kf_vals = np.array((10, 20, 30, 40, 80))

k_force = kf_vals[np.argmin(np.abs(kf_vals - kf_read))]
k_f = k_force / 2 
k_f *= 2 * np.pi
#print(k_force)

m_str = int(output_suffix.split('m_')[1].split('_')[0])
inv_str = int(output_suffix.split('inviscid_')[1].split('_')[0])
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])

dtype = np.complex128
dealias = 3/2
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dtype=dtype)

psi = dist.Field(name='psi', bases=disk)
vort = dist.Field(name='vort', bases=disk)

phi, r = dist.local_grids(disk)
phi_deal, r_deal = dist.local_grids(disk, scales=(dealias, dealias))

processed = np.load(file_str, allow_pickle=True)[()]

plt.figure()
plt.scatter(np.arange(processed['drifts_bad'].shape[0]), processed['drifts_bad'], s = 10, color = 'blue', marker = 'o')
plt.scatter(np.arange(processed['drifts_res'].shape[0]), processed['drifts_res'], s = 10, color = 'orange', marker = 'o')
plt.axhline(1e3, color = 'black', ls = 'dashed', label = 'currently chosen cut-off')
plt.yscale('log')
plt.ylabel(r'$1/(\delta_{j, \rm nearest})$')
plt.xlabel(r'mode index $j$')
plt.savefig('drifts_' + output_suffix + '.pdf')

print('saving figure: ' + 'drifts_' + output_suffix + '.pdf')

plt.figure()
plt.scatter(processed['evals_bad'].real, processed['evals_bad'].imag, s = 10, color = 'blue', marker = 'o')
plt.scatter(processed['evals_res'].real, processed['evals_res'].imag, s = 10, color = 'orange', marker = 'o')
plt.yscale('log')
plt.xscale('symlog')
plt.xlabel('Re')
plt.ylabel('Im')
plt.savefig('evals_' + output_suffix + '.pdf')

print('saving figure: ' + 'evals_' + output_suffix + '.pdf')

drift_sort = np.argsort(processed['drifts_res'])[::-1]
drifts_res_sorted = processed['drifts_res'][drift_sort]
evals_res_sorted = processed['evals_res'][drift_sort]
vort_evecs_res_sorted = processed['vort_evecs_res'][drift_sort, :, :]
n_modes = processed['drifts_res'].shape[0] 

x, y = coords.cartesian(phi_deal, r_deal)
cmap = 'RdBu_r'

output_path = pathlib.Path("evecs_" + output_suffix).absolute()
if not output_path.exists():
    output_path.mkdir()

print(output_suffix)
print(output_prefix)


for i in range(n_modes):
    print(i, output_path.joinpath("evecs_%d.png" %(i)))
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    vort.change_scales(dealias)
    vort['g'] = np.copy(vort_evecs_res_sorted[i, :, :])
    ax.pcolormesh(x, y, vort['g'].real, cmap=cmap)
    ax.set_title(r"$\omega_z$, $\omega =$" + "{:.3}".format(evals_res_sorted[i].real) + r"$+ i$" + "{:.3}".format(evals_res_sorted[i].imag))
    ax.set_axis_off()
    plt.tight_layout()
    
    plt.savefig(output_path.joinpath("evecs_%d.png" %(i)))
    plt.close()

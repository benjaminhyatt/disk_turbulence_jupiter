"""
Usage:
    plot_scalars.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the file being loaded [default: processed_scalars]
"""


import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)

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

processed = np.load(file_str, allow_pickle=True)[()]

plt.figure()
plt.plot(processed['t'], processed['EN']/k_f**2, color = 'orange', label = 'Z' + r'$/ k_f^2$')
plt.plot(processed['t'], processed['KE'], color = 'blue', label = 'K')
plt.xlabel('time')
plt.legend()
plt.title('Domain-averaged kinetic energy and enstrophy')
plt.yscale('log')
plt.tight_layout()
plt.savefig('ke_en_' + output_suffix + '_log.pdf')
print('saving figure: ' + 'ke_en_' + output_suffix + '_log.pdf')


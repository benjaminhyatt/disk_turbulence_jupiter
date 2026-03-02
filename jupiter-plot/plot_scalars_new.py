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
plt.savefig('ke_en_' + output_suffix + '.pdf')
print('saving figure: ' + 'ke_en_' + output_suffix + '.pdf')

plt.figure()
#plt.plot(processed['t'][-100:], processed['om0'][-100:], color = 'cyan', label = r'$\Omega_0*$') # extra factor of r
#plt.plot(processed['t'][-100:], processed['W'][-100:], color = 'blue', label = r'$\Omega_0$')
#plt.plot(processed['t'][-100:], processed['om1c'][-100:], color = 'orange', label = r'$\Omega_1$')
#plt.plot(processed['t'][-100:], processed['om1s'][-100:], color = 'red', label = r'$\tilde{\Omega}_1$')
plt.plot(processed['t'][-100:], processed['om1c'][-100:]/processed['W'][-100:], color = 'orange', label = r'$\Omega_1 / \Gamma $')
plt.plot(processed['t'][-100:], processed['om1s'][-100:]/processed['W'][-100:], color = 'red', label = r'$\tilde{\Omega}_1 / \Gamma$')
plt.plot(processed['t'][-100:], np.sqrt((processed['om1c'][-100:]/processed['W'][-100:])**2 + (processed['om1s'][-100:]/processed['W'][-100:])**2), color = 'purple')
plt.xlabel('time')
plt.legend(loc = 'lower left')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('trunc_' + output_suffix + '.pdf')
print('saving figure: ' + 'trunc_' + output_suffix + '.pdf')

#mock_integration = np.cumsum(np.diff(processed['t']) * (processed['nu0'] - alpha * processed['om0'])[:-1])
#mock_integration = np.concatenate(([processed['om0'][0]], processed['om0'][0] + mock_integration))

plt.figure()
plt.xlabel('time')
plt.plot(processed['t'], - alpha * processed['om0'], color = 'blue', label = 'friction')
plt.plot(processed['t'], processed['nu0'], color = 'orange', label = 'viscosity')
#plt.plot(processed['t'], mock_integration, color = 'green', linestyle = 'dashed', label = 'mock prediction') 
plt.plot(processed['t'], processed['om0'], color = 'black')
plt.xlim(0, 10)
plt.tight_layout()
plt.legend(loc = 'lower left')
plt.savefig('trunc_0_' + output_suffix + '.pdf')
print('saving figure: ' + 'trunc_0_' + output_suffix + '.pdf')





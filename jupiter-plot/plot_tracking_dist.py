"""
Usage:
    plot_tracking_dist.py <file>... [options]

Options:    
    --output=<str>               prefix in the name of the file being loaded [default: processed_tracking]
"""


import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from dedalus.extras import plot_tools
from scipy.stats import rice
import scipy
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

processed = np.load(file_str, allow_pickle=True)[()]

### radial pdf ###
r_centers = processed['r_centers']
r_edges = processed['r_edges']
drs = processed['drs']
pdf_r = processed['pdf_r']
plt.figure()
plt.bar(r_centers, pdf_r, drs, "Data")
plt.plot(r_centers, rice.pdf(r_centers, processed['rice_fit'][0], processed['rice_fit'][1], processed['rice_fit'][2]), color = "black", label = "Fit")
plt.xlabel(r'$r$')
plt.ylabel('Probability density')
plt.yscale('log')
plt.legend(loc = "upper right")
plt.ylim(0.5*np.min(1 / (processed['n_hist'] * drs)), 1e2)
plt.title('PDF (Fits: L=%e, sig=%e)' %(processed['rice_fit'][0] * processed['rice_fit'][2], processed['rice_fit'][2]))
plt.tight_layout()
plt.savefig('pdf_r_' + output_suffix + '.pdf')
print('saving figure: ' + 'pdf_r_' + output_suffix + '.pdf')

### radial hist ###
hist_r = processed['hist_r']
print(r_centers, hist_r)
n_hist = processed['n_hist']
rice_params = processed['rice_fit']
plt.figure()
plt.bar(r_centers, hist_r, drs, label = "Data")
#### plt.plot(r_centers, rice.pdf(r_centers, rice_params[0], rice_params[1], rice_params[2]), color = "black")
plt.scatter(r_centers, n_hist * drs * rice.pdf(r_centers, rice_params[0], rice_params[1], rice_params[2]), marker = '_', label = "Fit prediction")
plt.xlabel(r'$r$')
plt.ylabel('Counts')
plt.yscale('log')
plt.ylim(0.8, 1.1*np.max(hist_r))
plt.title('Histogram (Fits: L=%e, sig=%e)' %(rice_params[0] * rice_params[2], rice_params[2]))
plt.tight_layout()
plt.savefig('hist_r_' + output_suffix + '.pdf')
print('saving figure: ' + 'hist_r_' + output_suffix + '.pdf')


### phi pdf ###
phi_centers = processed['phi_centers']
dphis = processed['dphis']
pdf_phi = processed['pdf_phi']
plt.figure()
plt.bar(phi_centers, pdf_phi, dphis)
plt.xlabel(r'$\phi$')
plt.ylabel('Probability distribution')
plt.title('PDF')
plt.tight_layout()
plt.savefig('pdf_phi_' + output_suffix + '.pdf')
print('saving figure: ' + 'pdf_phi_' + output_suffix + '.pdf')

### phi hist ###
hist_phi = processed['hist_phi']
plt.figure()
plt.bar(phi_centers, hist_phi, dphis)
plt.xlabel(r'$\phi$')
plt.ylabel('Counts')
plt.title('Histogram')
plt.tight_layout()
plt.savefig('hist_phi_' + output_suffix + '.pdf')
print('saving figure: ' + 'hist_phi_' + output_suffix + '.pdf')


### 2d pdf ###
dpi = 300 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = dpi 

t_mar, b_mar, l_mar, r_mar = (0.15, 0.05, 0.05, 0.05)
h_plot, w_plot = (1., 1.) 
h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar
fig_width = 4.8 
scale = fig_width/w_total
fig = plt.figure(figsize = (scale * w_total, scale * h_total))

# construct ax 
left1 = (l_mar) / w_total
bottom1 = 1 - (t_mar + h_plot) / h_total
width1 = w_plot / w_total
height1 = h_plot / h_total
ax1 = fig.add_axes([left1, bottom1, width1, height1])

r_edges = processed['r_edges']
phi_edges_main = processed['phi_edges_main']
phi_edges_0a = processed['phi_edges_0a']
phi_edges_0b = processed['phi_edges_0b']
phi_edges = np.concatenate((phi_edges_main, [phi_edges_0b[1]]))
Phi, R = np.meshgrid(phi_edges, r_edges)
X = (R * np.cos(Phi)).T
Y = (R * np.sin(Phi)).T

pdf_2d = processed['pdf_2d']
ax1.pcolormesh(X, Y, pdf_2d, cmap = 'magma')
ax1.axis('off')
plt.savefig('pdf_2d_' + output_suffix + '.pdf')
plt.close()
print('saving figure: ' + 'pdf_2d_' + output_suffix + '.pdf')

### 2d hist ###
dpi = 300 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = dpi 

t_mar, b_mar, l_mar, r_mar = (0.15, 0.05, 0.05, 0.05)
h_plot, w_plot = (1., 1.) 
h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar
fig_width = 4.8 
scale = fig_width/w_total
fig = plt.figure(figsize = (scale * w_total, scale * h_total))

# construct ax 
left1 = (l_mar) / w_total
bottom1 = 1 - (t_mar + h_plot) / h_total
width1 = w_plot / w_total
height1 = h_plot / h_total
ax1 = fig.add_axes([left1, bottom1, width1, height1])

hist_2d = processed['hist_2d']
ax1.pcolormesh(X, Y, hist_2d, cmap = 'magma')
ax1.axis('off')
plt.savefig('hist_2d_' + output_suffix + '.pdf')
plt.close()
print('saving figure: ' + 'hist_2d_' + output_suffix + '.pdf')

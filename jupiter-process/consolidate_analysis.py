"""
This script loads in specified .h5 outputs, assumed to contain data from a simulation with one or more restarts,
and concatenates the data into a new numpy dictionary format for subsequent processing and/or plotting.

Usage:
    consolidate_analysis.py <files> [options]

Options:    
    --scalars=<bool>             output merged scalar tasks [default: True]
    --profiles=<bool>            output merged profile tasks [default: True]
    --snapshots=<bool>           output merged snapshot tasks [default: True]
    --output=<str>               prefix in the name of the merged output file [default: 'processed_analysis']
"""

import numpy as np
import h5py
from docopt import docopt
args = docopt(__doc__)

# initial timestep size
tstep = 1e-5

print("args read in")
print(args)

scalars = eval(args['--scalars'])
profiles = eval(args['--profiles'])
snapshots = eval(args['--snapshots'])
output_prefix = args['--output']

files = args['<files>']
if len(files[0]) == 1:
    files = [files]
output_suffix = files[0].split('analysis_')[1][:-1] 

t_out = np.array([])
processed = {}
if scalars:
    processed['scalars'] = {}
if profiles:
    processed['profiles'] = {}
if snapshots:
    processed['snapshots'] = {}

print('beginning loop over files')
for i, fi_str in enumerate(files): 
    fi = h5py.File(fi_str)
    print('file %i out of %i read in' %(i, len(files)))

    tasksi = list(fi['tasks'].keys())

    fi_t = fi['tasks/u'].dims[0]['sim_time'][:]

    if i > 0:
        idx_keep = np.where(fi_t > tf_last)[0][0]

    for task in tasksi:
        task_shape = np.array(fi['tasks'][task]).shape
        space_dim = np.sum(np.array(task_shape) > 1) - 1 
        if space_dim == 0 and scalars:
            if i > 0:
                processed['scalars'][task] = np.concatenate((processed['scalars'][task], np.copy(np.array(fi['tasks'][task][idx_keep:]))))
            else:
                processed['scalars'][task] = np.copy(np.array(fi['tasks'][task]))
        elif space_dim == 1 and profiles:
            if i > 0:
                processed['profiles'][task] = np.concatenate((processed['profiles'][task], np.copy(np.array(fi['tasks'][task][idx_keep:]))))
            else:
                processed['profiles'][task] = np.copy(np.array(fi['tasks'][task]))
        elif space_dim == 2 and snapshots:
            if i > 0:
                processed['snapshots'][task] = np.concatenate((processed['snapshots'][task], np.copy(np.array(fi['tasks'][task][idx_keep:]))))
            else:
                processed['snapshots'][task] = np.copy(np.array(fi['tasks'][task]))
    tf_last = np.copy(fi_t[-1])

print('saving merged files as processed_analysis_' + output_suffix + '.npy')
np.save(output_prefix + '_' + output_suffix + '.npy', processed)

"""
This script provides a virtually merged form of analysis data in specified .h5 files that contain data from a simulation with one or more restarts.
The merged data is concatenated in time, trimming out data from overlapping time segments.
Note: currently is set up for a 2d problem, and assumes that all analysis tasks' data are scalar-valued, except for snapshots, which may be vector-valued.
(Would need to modify the logic a bit (using the information found in dims) to treat more general cases)

Usage:
    virtual_merge.py <files>... [options]

Options:    
    --scalars=<bool>             output merged scalar tasks [default: True]
    --profiles=<bool>            output merged profile tasks [default: True]
    --snapshots=<bool>           output merged snapshot tasks [default: True]
    --output=<str>               prefix in the name of the merged output file [default: vm_analysis]
"""

import numpy as np
import h5py
from h5py import VirtualSource, VirtualLayout
import hashlib

from docopt import docopt
args = docopt(__doc__)

dtype = np.float64 

#print("args read in")
#print(args)

scalars = eval(args['--scalars'])
profiles = eval(args['--profiles'])
snapshots = eval(args['--snapshots'])
output_prefix = args['--output']

def get_int(string):
    string = string.split('.h5')[0]
    string = string.split('_s')[-1]
    integer = int(string)
    return integer

files = args['<files>']
if len(files[0]) == 1:
    files = [files]
    print("only read-in one file name argument - not recommended to use this script in that case")
files = sorted(files, key=get_int)
print('')
print('sorted files:')
for file_str in files:
    print(file_str)
nfiles = len(files)
output_suffix = files[0].split('analysis_')[1][:-1] 
Nphi = int(output_suffix.split('Nphi_')[1].split('_')[0])
Nr = int(output_suffix.split('Nr_')[1].split('_')[0])

# identify dataset characteristics
writes_read = {}
times_read = {}
tasks_read = {}
for i, fi_str in enumerate(files):
    with h5py.File(fi_str, 'r') as fi:
        # writes
        writes_read[i] = fi.attrs['writes']
        # times
        times_read[i] = np.copy(np.array(fi['tasks/u'].dims[0]['sim_time']))
        # tasks and scales
        tasks_read[i] = {}        
        for j, task in enumerate(list(fi['tasks'].keys())):
            tasks_read[i][task] = {}
            # attributes
            tasks_read[i][task]['attrs'] = []
            for attr in fi['tasks'][task].attrs:
                tasks_read[i][task]['attrs'].append(attr)
            tasks_read[i][task]['dims'] = {}
            for m, dim in enumerate(fi['tasks'][task].dims):
                tasks_read[i][task]['dims'][m] = {}
                tasks_read[i][task]['dims'][m]['label'] = dim.label
                tasks_read[i][task]['dims'][m]['name'] = []
                tasks_read[i][task]['dims'][m]['dtype'] = []
                if dim.label == 't':
                    for scalename in ['sim_time', 'wall_time', 'timestep', 'iteration', 'write_number']:
                        tasks_read[i][task]['dims'][m]['name'].append(scalename)
                        tasks_read[i][task]['dims'][m]['dtype'].append(np.array(fi['tasks'][task].dims[m][scalename]).dtype)
                else:
                    tasks_read[i][task]['dims'][m] = {}
                    tasks_read[i][task]['dims'][m]['label'] = dim.label
                    if dim.label == '' or dim.label == 'constant':
                        scalename = 'constant'
                    else:
                        hashval = hashlib.sha1(np.array(fi['tasks'][task].dims[m][0])).hexdigest()
                        scalename = dim.label + '_hash_' + hashval
                    tasks_read[i][task]['dims'][m]['name'] = scalename
                    tasks_read[i][task]['dims'][m]['dtype'] = np.array(fi['scales'][scalename]).dtype
            task_shape = np.array(fi['tasks'][task][0]).shape
            #tasks_read[i][task]['shape'] = task_shape
            space_dim = np.sum(np.array(task_shape) > 1) #- 1
            if space_dim == 0:
                tasks_read[i][task]['kind'] = 'scalar'
            elif space_dim == 1:
                #if np.array(task_shape)[2] > 1:
                if np.array(task_shape)[1] > 1:
                    tasks_read[i][task]['kind'] = 'profiler'
                #if np.array(task_shape)[1] > 1:
                elif np.array(task_shape)[0] > 1:
                    tasks_read[i][task]['kind'] = 'profilephi'
            elif space_dim == 2:
                if np.array(task_shape)[0] == 2:
                    if np.array(task_shape)[2] > 1:
                        tasks_read[i][task]['kind'] = 'profiler_field'
                    elif np.array(task_shape)[1] > 1:
                        tasks_read[i][task]['kind'] = 'profilephi_field'
                else:
                    tasks_read[i][task]['kind'] = 'snapshot_scalar'
            elif space_dim == 3:
                tasks_read[i][task]['kind'] = 'snapshot_field'

# merging decisions
print("merging decisions")
writes_merge = 0
ends_indiv = np.empty(nfiles, dtype=np.int_)
starts_merge = np.empty(nfiles, dtype=np.int_)
ends_merge = np.empty(nfiles, dtype=np.int_)
for j in range(nfiles):
    i = nfiles - j - 1
    if j == 0:
        ends_indiv[i] = times_read[i].shape[0] - 1
    else:
        ends_indiv[i] = np.where(times_read[i] < times_read[i + 1][0])[0][-1]
for i in range(nfiles):
    if i == 0:
        starts_merge[i] = 0
        ends_merge[i] = ends_indiv[i]
    else:
        starts_merge[i] = ends_merge[i - 1] + 1
        ends_merge[i] = ends_merge[i - 1] + 1 + ends_indiv[i]
    writes_merge += ends_merge[i] + 1 - starts_merge[i]


# perform merging
print('saving merged files as ' + output_prefix + '_' + output_suffix + '.h5')
with h5py.File(output_prefix + '_' + output_suffix + '.h5', 'w') as f_merge:
    ntasks = len(list(tasks_read[0].keys()))
    f_merge.attrs['writes'] = writes_merge
    f_merge.create_group('scales')
    
    # link tasks virtually
    for k, task in enumerate(list(tasks_read[0].keys())):
        if tasks_read[0][task]['kind'] == 'scalar' and scalars:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, 1, 1), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, 0, 0] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, 1, 1)) 
        elif tasks_read[0][task]['kind'] == 'profiler' and profiles:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, 1, Nr), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, 0, :] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, 1, Nr))
        elif tasks_read[0][task]['kind'] == 'profilephi' and profiles:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, Nphi, 1), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, :, 0] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, Nphi, 1))
        elif tasks_read[0][task]['kind'] == 'profiler_field' and profiles:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, 2, 1, Nr), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, :, 0, :] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, 2, 1, Nr))
        elif tasks_read[0][task]['kind'] == 'profilephi_field' and profiles:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, 2, Nphi, 1), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, :, :, 0] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, 2, Nphi, 1))  
        elif tasks_read[0][task]['kind'] == 'snapshot_field' and snapshots:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, 2, Nphi, Nr), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, :, :, :] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, 2, Nphi, Nr)) 
        elif tasks_read[0][task]['kind'] == 'snapshot_scalar' and snapshots:
            merge_layout = VirtualLayout(shape=(ends_merge[nfiles - 1] + 1, Nphi, Nr), dtype=dtype)
            for i in range(nfiles):
                merge_layout[starts_merge[i]:ends_merge[i] + 1, :, :] = VirtualSource(files[i], 'tasks/' + task, shape = (ends_indiv[i] + 1, Nphi, Nr))
        f_merge.create_virtual_dataset('tasks/' + task, merge_layout)
    
    # hard copy scales (if label is 't' copy from each file, otherwise just once)
    for i, fi_str in enumerate(files):
        with h5py.File(fi_str, 'r') as fi:
            for k, task in enumerate(list(tasks_read[0].keys())):
                #for attr in tasks_read[0][task]['attrs']:
                #    if (not (attr == 'scales')) and (attr not in list(f_merge['tasks'][task].attrs.keys())):
                #        print(attr)
                #        f_merge['tasks'][task].attrs[attr] = fi['tasks'][task].attrs[attr]
                f_merge['tasks'][task].attrs['grid_space'] = fi['tasks'][task].attrs['grid_space']
                for m in list(tasks_read[0][task]['dims'].keys()):
                    #print(m, task, list(f_merge['tasks'][task].dims))
                    f_merge['tasks'][task].dims[m].label = tasks_read[0][task]['dims'][m]['label']
                    if tasks_read[0][task]['dims'][m]['label'] == 't':
                        for scalename in tasks_read[0][task]['dims'][m]['name']:
                            scalename_use = scalename + '_' + str(i)
                            if scalename_use not in list(f_merge['scales'].keys()):
                                # hard copy
                                fi.copy('scales/' + scalename, f_merge['scales'], name = scalename_use)
                    else:
                        scalename = tasks_read[0][task]['dims'][m]['name']
                        if 'hash' in scalename: 
                            scalename_use = tasks_read[0][task]['dims'][m]['label']
                        else:
                            scalename_use = scalename 
                        if scalename_use not in list(f_merge['scales'].keys()):
                            fi.copy('scales/' + scalename, f_merge['scales'], name = scalename_use)
    
    # merge hard-copied scales (label = 't' only)
    for k, task in enumerate(list(tasks_read[0].keys())):
        for m in list(tasks_read[0][task]['dims'].keys()):
            if tasks_read[0][task]['dims'][m]['label'] == 't':
                for s, scalename in enumerate(tasks_read[0][task]['dims'][m]['name']):
                    if scalename not in list(f_merge['scales'].keys()):
                        f_merge['scales'].create_dataset(scalename, shape=(ends_merge[nfiles - 1] + 1,), dtype=tasks_read[0][task]['dims'][m]['dtype'][s])
                        for i in range(nfiles):
                            scalename_to_merge = scalename + '_' + str(i)
                            f_merge['scales'][scalename][starts_merge[i]:ends_merge[i] + 1] = f_merge['scales'][scalename_to_merge][0:ends_indiv[i] + 1]

    # make/attach scales to tasks
    for k, task in enumerate(list(tasks_read[0].keys())):
        for m in list(tasks_read[0][task]['dims'].keys()):
            if tasks_read[0][task]['dims'][m]['label'] == 't':
                for scalename in tasks_read[0][task]['dims'][m]['name']:
                    f_merge['scales'][scalename].make_scale(scalename)
                    f_merge['tasks'][task].dims[m].attach_scale(f_merge['scales'][scalename])
            else:
                scalename = tasks_read[0][task]['dims'][m]['name']
                if 'hash' in scalename:
                    scalename_use = tasks_read[0][task]['dims'][m]['label']
                else:
                    scalename_use = scalename
                f_merge['scales'][scalename_use].make_scale(scalename_use)
                f_merge['tasks'][task].dims[m].attach_scale(f_merge['scales'][scalename_use])

print("finishing")

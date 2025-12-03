import numpy as np
from mpi4py import MPI 
import logging
logger = logging.getLogger(__name__.split('.')[-1])
MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size

import re

def natural_sort(iterable, reverse=False):
    """ 
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Based on http://stackoverflow.com/a/4836734.

    """

    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)

def visit_writes(set_paths, function, **kw):
    """
    Apply function to writes from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    function : function(set_path, start, count, **kw)
        A function on an HDF5 file, start index, and count.

    Other keyword arguments are passed on to `function`

    Notes
    -----
    This function is parallelized over writes, and so can be effectively
    parallelized up to the number of writes from all specified sets.

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    arg_list = zip(set_paths, *get_assigned_writes(set_paths))
    for set_path, start, count in arg_list:
        if count:
            logger.info("Visiting set {} (start: {}, end: {})".format(set_path, start, start+count))
            function(set_path, start, count, **kw)


def get_assigned_writes(set_paths):
    """
    Divide writes from a list of analysis sets between MPI processes.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths
    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    # Distribute all writes in blocks
    writes = get_all_writes(set_paths)
    block = int(np.ceil(sum(writes) / MPI_SIZE))
    proc_start = MPI_RANK * block
    # Find set start/end indices
    set_ends = np.cumsum(writes)
    set_starts = set_ends - writes
    # Find proc start indices and counts for each set
    starts = np.clip(proc_start, a_min=set_starts, a_max=set_ends)
    counts = np.clip(proc_start+block, a_min=set_starts, a_max=set_ends) - starts
    return starts-set_starts, counts

def get_all_writes(set_paths):
    """
    Get write numbers from a list of analysis sets.

    Parameters
    ----------
    set_paths : list of str or pathlib.Path
        List of set paths

    """
    set_paths = natural_sort(str(sp) for sp in set_paths)
    writes = []
    for set_path in set_paths:
        f = np.load(str(set_path), allow_pickle=True)[()]
        try:
            writes.append(f['nout'])
        except:
            writes.append(len(f['ws']))
    return writes

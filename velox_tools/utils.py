# velox_tools/utils.py
"""Small helpers: timing, a dask cluster and outlier masking."""
import time
from functools import wraps

import numpy as np


def timing_wrapper(func):
    """Decorator that prints the runtime of every call of `func`.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        `func` with the same signature and docstring.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Executed {func.__name__} in {end - start:.4f} seconds")
        return result
    return wrapper


def make_cluster(n_workers=16, threads_per_worker=1, memory_limit='16GB',
                  dashboard_address=':8787', local_directory=None, processes=True):
    """Start a local dask cluster and connect a client to it.

    Prints the address of the dashboard. On a shared machine, port 8787 is
    often taken; dask then picks another one, so read the printed link.

    Parameters
    ----------
    n_workers : int, default 16
        Number of workers.
    threads_per_worker : int, default 1
        Threads per worker.
    memory_limit : str, default '16GB'
        Memory limit per worker.
    dashboard_address : str, default ':8787'
        Address of the dashboard.
    local_directory : str, optional
        Scratch directory for spilling to disk. Point it at a disk with
        enough space if ``/tmp`` is small.
    processes : bool, default True
        Use worker processes. In a plain script (not a notebook), the call
        then has to sit inside ``if __name__ == '__main__':``. False gives
        threaded workers, which need no guard but only run in parallel
        where the code releases the GIL (numpy mostly does).

    Returns
    -------
    client : dask.distributed.Client
    cluster : dask.distributed.LocalCluster
    """
    from dask.distributed import Client, LocalCluster

    kwargs = dict(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        processes=processes,
    )
    if local_directory is not None:
        kwargs['local_directory'] = local_directory
    cluster = LocalCluster(**kwargs)
    client = Client(cluster)
    print(f"dashboard: {client.dashboard_link}")
    return client, cluster


def mask_percentile_outliers(data, lo=1, hi=99):
    """Set values outside a percentile range to NaN.

    Parameters
    ----------
    data : array-like
        Input values.
    lo, hi : float, default 1, 99
        Percentiles (0-100) of the range to keep, ignoring NaNs.

    Returns
    -------
    numpy.ndarray of float64
        Copy of `data` with values below the `lo` or above the `hi`
        percentile set to NaN.
    """
    data = np.asarray(data, dtype='float64')
    lo_val, hi_val = np.nanpercentile(data, [lo, hi])
    return np.where((data < lo_val) | (data > hi_val), np.nan, data)

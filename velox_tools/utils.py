# velox_tools/utils.py

import time
from functools import wraps

import numpy as np


def timing_wrapper(func):
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
    """Start a dask LocalCluster + Client with sane VELOX-scale defaults.

    The same ~6 lines (LocalCluster(...) + Client(cluster) + print the
    dashboard link) were copy-pasted into most notebooks that touch a full
    campaign archive. Worth knowing before you call this:

    - `processes=True` (default) workers need the call wrapped in
      `if __name__ == '__main__':` when called from a plain .py script
      (not a notebook) -- they re-import the calling module to spawn, and
      without the guard that re-runs the whole script and crashes with a
      RuntimeError about "safe importing of main module". Pass
      `processes=False` for threaded workers instead (no spawn, no guard
      needed, but only genuinely parallel for code that releases the GIL
      -- numpy/scipy internals mostly do).
    - `local_directory` should point at real scratch space, not /tmp --
      some hosts (e.g. passat) have little to no space there and dask's
      disk-based shuffle will fail with P2POutOfDiskError if it spills.
    - Port 8787 is often already taken by someone else's cluster on a
      shared host; dask will silently pick another port and this prints
      whatever it actually bound to -- read that, don't assume 8787.

    Returns
    -------
    (client, cluster)
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
    """Replace values outside the [lo, hi] percentile range with NaN.

    Common first step on raw VELOX ASCII/calibration data before any
    further processing (drops a handful of extreme outlier pixels/frames
    that would otherwise dominate a colorbar or a mean).
    """
    data = np.asarray(data, dtype='float64')
    lo_val, hi_val = np.nanpercentile(data, [lo, hi])
    return np.where((data < lo_val) | (data > hi_val), np.nan, data)

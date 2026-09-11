# velox_tools/io.py
"""Loaders for VELOX housekeeping data.

For the brightness temperatures themselves, see
:func:`velox_tools.campaign.load_velox`.
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd
import xarray as xr

from velox_tools.config import load_config

_DATE_RE = re.compile(r'(\d{8})')


def _date_from_path(path: str) -> str:
    """First YYYYMMDD date in a file path.

    The T3/T4 logs carry no date of their own; it comes from the flight
    directory (e.g. ``.../HALO-AC3_20220320_HALO_RF07/.../Additional/T4.txt``).
    """
    m = _DATE_RE.search(path)
    if not m:
        raise ValueError(f"couldn't find a YYYYMMDD date in path: {path}")
    return m.group(1)


def load_instrument_temperatures(base_glob: str | None = None) -> tuple[xr.Dataset, xr.Dataset]:
    """Lens (T3) and germanium-window (T4) temperatures of VELOX.

    Reads the ``T3*.txt`` and ``T4.txt`` logs of all flights matching
    `base_glob` and concatenates them.

    Parameters
    ----------
    base_glob : str, optional
        Glob pattern of the directories holding the logs. Default: the
        ``Additional`` directories of all HALO-(AC)3 flights below
        ``data_root`` (see :mod:`velox_tools.config`).

    Returns
    -------
    T3, T4 : xarray.Dataset
        Indexed by ``time`` (sorted, without duplicates), with the variable
        ``T3`` or ``T4``. Empty if no files were found for that sensor.

    Raises
    ------
    FileNotFoundError
        If neither T3 nor T4 files match `base_glob`.
    """
    if base_glob is None:
        base_glob = os.path.join(load_config().data_root,
                                 'HALO-AC3/02_Flights/HALO-AC3_*/VELOX/VELOX_327kveL/Processed/Additional')
    t3_files = glob.glob(f'{base_glob}/T3*.txt')
    t4_files = glob.glob(f'{base_glob}/T4.txt')
    if not t3_files and not t4_files:
        raise FileNotFoundError(f"no T3*.txt or T4.txt files found under {base_glob!r}")

    t4_parts = []
    for f in t4_files:
        df = pd.read_csv(f, sep=r'\s+', names=['SOD', 'T4'], skiprows=10)
        date = _date_from_path(f)
        df['time'] = pd.to_datetime(date, format='%Y%m%d') + pd.to_timedelta(df['SOD'], unit='s')
        df = df.drop(columns=['SOD'])
        t4_parts.append(df.set_index('time').to_xarray())
    t4 = xr.concat(t4_parts, dim='time').sortby('time').drop_duplicates('time') if t4_parts else xr.Dataset()

    t3_parts = []
    for f in t3_files:
        df = pd.read_csv(f, sep=r'\s+', names=['index_', 'T3', 'SOD'], skiprows=10)
        date = _date_from_path(f)
        df['time'] = pd.to_datetime(date, format='%Y%m%d') + pd.to_timedelta(df['SOD'], unit='s')
        df = df.drop(columns=['SOD', 'index_'])
        t3_parts.append(df.set_index('time').to_xarray())
    t3 = xr.concat(t3_parts, dim='time').sortby('time').drop_duplicates('time') if t3_parts else xr.Dataset()

    return t3, t4

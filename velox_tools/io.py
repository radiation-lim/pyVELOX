# velox_tools/io.py
"""Loaders for VELOX ancillary instrument data (not the BT_2D imagery
itself -- see the processed netCDF/zarr archives for that)."""
from __future__ import annotations

import glob
import re

import pandas as pd
import xarray as xr

_DATE_RE = re.compile(r'(\d{8})')


def _date_from_path(path: str) -> str:
    """Pull the first 8-digit YYYYMMDD run of digits out of a file path.

    VELOX Additional/T3*.txt, T4.txt files carry no date of their own --
    it has to come from the campaign flight-day directory name (e.g.
    .../HALO-AC3_20220320_HALO_RF07/VELOX/.../Additional/T4.txt). Matching
    by regex instead of a fixed `path.split('/')[-N]` index (as the
    original notebooks did) is robust to the exact directory depth
    varying between campaigns/setups.
    """
    m = _DATE_RE.search(path)
    if not m:
        raise ValueError(f"couldn't find a YYYYMMDD date in path: {path}")
    return m.group(1)


def load_instrument_temperatures(
    base_glob: str = '/projekt_agmwend/data/HALO-AC3/02_Flights/HALO-AC3_*/VELOX/VELOX_327kveL/Processed/Additional',
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load and concatenate VELOX lens (T3) and germanium-window (T4)
    temperature logs across all flight days matching `base_glob`.

    Same logic as was copy-pasted into desperate_correction.ipynb,
    desperate_prepare_df_for_correction.ipynb, and
    desperate_correction_application.ipynb.

    Returns
    -------
    (T3, T4) : xr.Dataset, xr.Dataset
        Each indexed by `time`, sorted, de-duplicated. T3 has data var
        'T3' (lens temperature), T4 has 'T4' (germanium-window
        temperature). Either may be an empty Dataset if no matching files
        were found for that instrument.
    """
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

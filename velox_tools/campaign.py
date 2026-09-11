# velox_tools/campaign.py
"""Time-based access to the final VELOX data of HALO-(AC)3 and PERCUSION.

Every function takes a time (a ``slice``, a timestamp or a date string);
campaign and research flight follow from it (see :func:`flight`). The
module finds the final VELOX brightness-temperature files and the BAHAMAS
navigation data of that flight below ``data_root`` (see
:mod:`velox_tools.config`) and builds on :mod:`velox_tools.processing` and
:mod:`velox_tools.georef_paulr`::

    from velox_tools import campaign

    ds = campaign.load_velox(slice('2024-08-25T12:00', '2024-08-25T12:10'))
    pb = campaign.pushbroom(slice('2022-03-20T10:00', '2022-03-20T10:30'), georef=True)
    geo = campaign.georef(slice('2024-08-25T12:00:00', '2024-08-25T12:00:10'), channel=3)

File paths below ``data_root`` follow the layout of the campaign archive
(see :data:`CAMPAIGNS`).
"""
import glob
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from velox_tools import georef_paulr, processing
from velox_tools.config import load_config

#: Channel numbers of the final data: 1 (7.7-12.0 um), 2 (8.65 um),
#: 3 (10.74 um), 5 (11.66 um) and 6 (12.0 um).
CHANNELS = [1, 2, 3, 5, 6]

#: Per campaign: file patterns of the final VELOX and BAHAMAS files relative
#: to ``data_root``, the dask chunks to open the VELOX files with, and the
#: flight ID (date) of every research flight.
CAMPAIGNS = {
    'HALO-AC3': dict(
        velox='HALO-AC3/02_Flights/HALO-AC3_{flight}_HALO_{rf}/VELOX/VELOX_327kveL/Processed/to_publish/final/'
              'HALO-AC3_HALO_VELOX_BT{channel}_*_{flight}_{rf}_v3.0.nc',
        bahamas='HALO-AC3/02_Flights/HALO-AC3_{flight}_HALO_{rf}/BAHAMAS/HALO-AC3_HALO_BAHAMAS_{flight}_{rf}_v1.nc',
        chunks={},  # native netCDF chunks (2222, 46, 37)
        flights=dict(RF03='20220313', RF04='20220314', RF05='20220315', RF06='20220316',
                     RF07='20220320', RF08='20220321', RF09='20220328', RF10='20220329',
                     RF11='20220330', RF12='20220401', RF13='20220404', RF14='20220407',
                     RF15='20220408', RF16='20220410', RF17='20220411', RF18='20220412'),
    ),
    'PERCUSION': dict(
        velox='PERCUSION/01_Flights/PERCUSION_{flight}/VELOX/06_final_files/PERCUSION_HALO_VELOX_BT{channel}_*_{flight}.nc',
        # BAHAMAS letter doesn't always match the folder (HALO-20240827b in PERCUSION_20240827a)
        bahamas='PERCUSION/01_Flights/PERCUSION_{flight}/BAHAMAS/HALO-*_BAHAMAS_V01.nc',
        chunks={'time': 100},  # files are contiguous
        flights=dict(RF01='20240811a', RF02='20240813a', RF03='20240816a', RF04='20240818a',
                     RF05='20240821a', RF06='20240822a', RF07='20240825a', RF08='20240827a',
                     RF09='20240829a', RF10='20240831a', RF11='20240903a', RF12='20240906a',
                     RF13='20240907a', RF14='20240909a', RF15='20240912a', RF16='20240914a',
                     RF17='20240916a', RF18='20240919a', RF19='20240921a', RF20='20240923a',
                     RF21='20240924a', RF22='20240926a', RF23='20240928a',
                     ECVal_RF01='20241105a', ECVal_RF02='20241107a', ECVal_RF03='20241110a',
                     ECVal_RF04='20241112a', ECVal_RF05='20241114a', ECVal_RF06='20241116a'),
    ),
}

_BAHAMAS = dict(IRS_LAT='lat', IRS_LON='lon', IRS_ALT='alt', IRS_GS='gs',
                IRS_HDG='hdg', IRS_PHI='roll', IRS_THE='pitch')


def flight(time):
    """Campaign and research flight at a given time.

    Parameters
    ----------
    time : slice, str or datetime-like
        A time slice, a timestamp or a date string. Only the (start) date is
        used -- every research flight is on a different day.

    Returns
    -------
    tuple of str
        ``(campaign, research_flight)``, e.g. ``('PERCUSION', 'RF07')``.

    Raises
    ------
    ValueError
        If no research flight with final VELOX data took place on that day.
    """
    start = pd.Timestamp(time.start if isinstance(time, slice) else time)
    for campaign, c in CAMPAIGNS.items():
        for rf, flight_id in c['flights'].items():
            if flight_id.startswith(f'{start:%Y%m%d}'):
                return campaign, rf
    raise ValueError(f"no research flight with final VELOX data on {start:%Y-%m-%d}")


def _path(campaign, rf, kind, **kwargs):
    c = CAMPAIGNS[campaign]
    pattern = c[kind].format(flight=c['flights'][rf], rf=rf, **kwargs)
    config = load_config()
    files = sorted(glob.glob(os.path.join(config.data_root, pattern)))
    if not files:
        source = f"from {config.config_file}" if config.config_file else f"no config.yaml found from {os.getcwd()}"
        raise FileNotFoundError(
            f"{pattern} not found under data_root={config.data_root!r} ({source}) -- if the server "
            f"is mounted elsewhere, set data_root in config.yaml or VELOX_DATA_ROOT (e.g. 'P:/data')"
        )
    return files[0]


@lru_cache(maxsize=8)
def _load_nav(campaign, rf):
    ds = xr.open_dataset(_path(campaign, rf, 'bahamas'))
    ds = ds.set_coords('TIME').swap_dims(tid='TIME')[list(_BAHAMAS)]
    return ds.rename(TIME='time', **_BAHAMAS).load()


def load_nav(time):
    """BAHAMAS navigation data of the research flight at `time`.

    Parameters
    ----------
    time : slice, str or datetime-like
        Any time during the flight (see :func:`flight`).

    Returns
    -------
    xarray.Dataset
        10 Hz data of the whole flight, indexed by ``time``, with the
        variables ``lat``, ``lon``, ``alt``, ``gs``, ``hdg``, ``roll`` and
        ``pitch`` -- the naming used throughout this package. Cached, so
        repeated calls for the same flight don't re-read the file.
    """
    return _load_nav(*flight(time))


def interp_nav(nav, time):
    """Interpolate navigation data onto other time steps.

    Linear in time for all variables; the heading is interpolated via its
    sine and cosine, so 359 and 1 deg average to 0 and not to 180 deg.

    Parameters
    ----------
    nav : xarray.Dataset
        Navigation data indexed by ``time`` with a heading variable ``hdg``
        (e.g. from :func:`load_nav`).
    time : array-like or xarray.DataArray
        Target times, e.g. the frame times of a VELOX dataset.

    Returns
    -------
    xarray.Dataset
        `nav` at `time`.
    """
    out = nav.interp(time=time)
    hdg = np.deg2rad(nav['hdg'])
    out['hdg'] = np.rad2deg(np.arctan2(np.sin(hdg).interp(time=time), np.cos(hdg).interp(time=time))) % 360
    return out


def load_velox(time, channels=CHANNELS):
    """Final VELOX brightness temperatures of one or several channels.

    The filter wheel records each channel at a slightly different time
    (PERCUSION: 10-50 ms apart; HALO-AC3: different frames filtered out per
    channel), so all channels are put on the frame times of the first one
    (nearest frame within 0.5 s, NaN where a channel has none).

    Parameters
    ----------
    time : slice or str
        Time slice, or a date string for the whole flight.
    channels : int or list of int, default :data:`CHANNELS`
        Channel number(s) to load.

    Returns
    -------
    xarray.Dataset
        Lazy (dask-backed) dataset with ``BT_2D`` (band, time, x, y) in
        degC on the 635 x 507 pixel grid and ``BT_Center`` (band, time), the
        mean of the central 10 x 10 pixels. The ``band`` coordinate holds
        the channel numbers; ``campaign`` and ``research_flight`` are stored
        as attributes. Flight direction is towards +y, x = 0 is on the
        starboard side.

    Notes
    -----
    The HALO-AC3 files are chunked 2222 frames deep, so even a few seconds
    of data read ~3 GB per channel. Over a network mount, cut out one
    longer segment rather than many short ones.
    """
    campaign, rf = flight(time)
    channels = np.atleast_1d(channels).tolist()
    bands = []
    for channel in channels:
        ds = xr.open_dataset(_path(campaign, rf, 'velox', channel=channel), chunks=CAMPAIGNS[campaign]['chunks'])
        ds = ds.rename({'BT_center': 'BT_Center'}) if 'BT_center' in ds else ds  # PERCUSION
        ds = ds[['BT_2D', 'BT_Center']].transpose('time', 'x', 'y')  # PERCUSION is (time, y, x)
        ds = ds.sel(time=time)
        if bands:
            ds = ds.reindex(time=bands[0].time, method='nearest', tolerance=np.timedelta64(500, 'ms'))
        bands.append(ds)

    ds = xr.concat(bands, dim='band').assign_coords(band=channels, x=np.arange(635), y=np.arange(507))
    ds.attrs = dict(campaign=campaign, research_flight=rf)
    return ds


def _load(time, channels):
    nav = load_nav(time)
    ds = load_velox(time, channels)
    # VELOX can start before BAHAMAS (e.g. HALO-AC3 RF07)
    ds = ds.sel(time=slice(nav.time.values[0], nav.time.values[-1]))
    return ds, nav


def _georef_series(ds, nav, channel, **kwargs):
    geo = georef_paulr.georef_series(interp_nav(nav, ds.time), channel, ds.attrs['campaign'], footprint=True, **kwargs)
    return geo.rename({'x-pixel': 'x', 'y-pixel': 'y', 'lat': 'lats', 'lon': 'lons'})


def pushbroom(time, channels=CHANNELS, slicing_position=250, georef=False, **kwargs):
    """Pushbroom image of a flight segment.

    Loads VELOX and BAHAMAS data for `time` and passes them to
    :func:`velox_tools.processing.pushbroom`.

    Parameters
    ----------
    time : slice or str
        Time slice, or a date string for the whole flight.
    channels : int or list of int, default :data:`CHANNELS`
        Channel number(s) to include.
    slicing_position : int, default 250
        First image row (y) of the strip cut out of every frame.
    georef : bool, default False
        If True, also return ``lats``/``lons`` (y, time) for every pushbroom
        pixel. Only the strip rows that go into the image are georeferenced
        (with :func:`velox_tools.georef_paulr.georef_series`, using the
        geometry of the first channel -- the channels are footprint-aligned)
        and then pushbroomed along with ``BT_2D``.
    **kwargs
        Passed to :func:`velox_tools.processing.pushbroom`, e.g.
        ``time_correction`` or ``quality_flag``.

    Returns
    -------
    xarray.Dataset
        ``BT_2D`` (band, y, time), plus ``lats``/``lons`` (y, time) with
        ``georef=True``. See :func:`velox_tools.processing.pushbroom`.
    """
    ds, nav = _load(time, channels)
    if georef:
        pixel_per_second = processing.compute_pixel_per_second(nav.interp(time=ds.time))
        rows = np.arange(slicing_position, min(slicing_position + pixel_per_second.max(), 507))
        geo = _georef_series(ds, nav, int(ds.band[0]), rows=rows)
        # as dask, filling the other rows with NaN (aligning to ds) stays lazy
        ds = ds.assign(lats=geo['lats'].chunk(), lons=geo['lons'].chunk())
    return processing.pushbroom(ds, slicing_position=slicing_position, nav_data=nav, **kwargs)


def georef(time, channel=3, method='mounttree', **kwargs):
    """Brightness temperatures of one channel with per-pixel coordinates.

    Parameters
    ----------
    time : slice or str
        Time slice. Georeferencing costs a fraction of a second per frame,
        so keep it short.
    channel : int, default 3
        Channel number.
    method : {'mounttree', 'project'}, default 'mounttree'
        ``'mounttree'``: per-pixel view-direction calibration and
        per-channel boresight offsets
        (:func:`velox_tools.georef_paulr.georef_series`).
        ``'project'``: shared analytic pinhole model
        (:func:`velox_tools.processing.project`).
    **kwargs
        With ``method='mounttree'``, passed to
        :func:`velox_tools.georef_paulr.georef_series` (e.g.
        ``ref_height``, ``flat_earth``).

    Returns
    -------
    xarray.Dataset
        ``BT_2D`` and ``BT_Center`` of the channel with ``lats``/``lons``
        (time, x, y). ``'mounttree'`` also returns the ``height`` of each
        pixel; ``'project'`` adds the variables listed in
        :func:`velox_tools.processing.project`.
    """
    ds, nav = _load(time, channel)
    ds = ds.isel(band=0)
    if method == 'project':
        return processing.project(ds, nav_data=interp_nav(nav, ds.time))
    return ds.merge(_georef_series(ds, nav, channel, **kwargs))

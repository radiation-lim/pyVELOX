# velox_tools/georef_paulr.py
"""Per-pixel georeferencing of VELOX frames with calibrated view directions.

The more precise of the two georeferencing options (the other is
:func:`velox_tools.processing.project`):

- Every pixel has its own calibrated view direction (view-direction
  cosines, one calibration file per channel, shipped with the package).
- Every channel has its own boresight offset angles -- the five channels
  point in slightly different directions (up to ~0.5 deg apart), with
  separate calibrations for HALO-(AC)3 and PERCUSION.
- The view directions are intersected iteratively with a plane at a given
  height above the WGS84 ellipsoid.

The coordinate transformations use the `mounttree
<https://pypi.org/project/mounttree/>`_ package. Ported from the per-pixel
georeferencing script ``Velox_GeoRef_EachPx.py`` by Paul R.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import xarray as xr
import yaml

try:
    import mounttree as mnt
except ImportError as _e:  # pragma: no cover - exercised only when mounttree is missing
    mnt = None
    _MOUNTTREE_IMPORT_ERROR = _e
else:
    _MOUNTTREE_IMPORT_ERROR = None


_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'georef_paulr')
_CONFIG_PATH = os.path.join(_DATA_DIR, 'velox_mounttree.yaml')

#: Offset (x0, y0) of each channel's 635 x 507 window of the final data on
#: the 640 x 512 sensor (channel footprint alignment of the L1 processing,
#: the same for HALO-(AC)3 and PERCUSION).
FOOTPRINT = {1: (1, 3), 2: (1, 3), 3: (5, 1), 5: (0, 0), 6: (4, 5)}


def _require_mounttree():
    if mnt is None:
        raise ImportError(
            "georef_paulr requires the 'mounttree' package (pip install mounttree)"
        ) from _MOUNTTREE_IMPORT_ERROR


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def available_channels(campaign: str = 'HALO-AC3') -> list[str]:
    """Channels with a boresight calibration for a campaign.

    Parameters
    ----------
    campaign : {'HALO-AC3', 'PERCUSION'}
        Campaign name.

    Returns
    -------
    list of str
        E.g. ``['Channel1', 'Channel2', 'Channel3', 'Channel5', 'Channel6']``.

    Raises
    ------
    ValueError
        If there is no calibration for `campaign`.
    """
    config = _load_config()
    try:
        return sorted(config['campaign'][campaign].keys())
    except KeyError:
        raise ValueError(
            f"No calibration for campaign={campaign!r}. "
            f"Available campaigns: {sorted(_load_config()['campaign'])}"
        )


def _channel_calibration(channel: int, campaign: str):
    config = _load_config()
    key = f'Channel{channel}'
    try:
        entry = config['campaign'][campaign][key]
    except KeyError:
        raise ValueError(
            f"No calibration for campaign={campaign!r} channel={channel!r}. "
            f"Try velox_tools.georef_paulr.available_channels({campaign!r})"
        )
    vdc_path = os.path.join(_DATA_DIR, entry['VDC-File'])
    return entry['Offset-Angles'], vdc_path


@dataclass
class _ChannelGeometry:
    """Everything about a channel's geometry that does NOT depend on
    aircraft position/attitude -- loaded once from disk, reused across
    every frame in a series instead of re-reading the VDC file and
    rebuilding the coordinate tree on every timestep (which is what made
    a naive per-frame loop take ~10s/frame)."""
    coord_sys: object  # mnt.CoordinateUniverse
    offset_angles: dict
    vd_vector_velox: np.ndarray  # (3, nx, ny), view directions in the VELOX frame
    x_pixel: np.ndarray
    y_pixel: np.ndarray


def _load_channel_geometry(channel: int, campaign: str, footprint: bool = False) -> _ChannelGeometry:
    _require_mounttree()
    offset_angles, vdc_path = _channel_calibration(channel, campaign)

    vdc = xr.open_dataset(vdc_path)
    zenith = np.deg2rad(vdc['zenith'].values)
    azimuth = np.deg2rad(vdc['azimuth'].values)
    vd_vector = np.stack((
        np.tan(zenith) * np.cos(azimuth),
        -np.tan(zenith) * np.sin(azimuth),
        np.ones(zenith.shape),
    ))
    x_pixel, y_pixel = vdc['x-pixel'].values, vdc['y-pixel'].values
    vdc.close()

    if footprint:
        # the VDC grid is rotated 180 deg against the final images (VDC: y=0
        # forward, x=0 port; images: flight towards +y, x=0 starboard) --
        # checked 2026-09-11 by frame-to-frame overlap in turns, both campaigns
        x0, y0 = FOOTPRINT[channel]
        vd_vector = vd_vector[:, ::-1, ::-1][:, x0:x0 + 635, y0:y0 + 507]
        x_pixel, y_pixel = np.arange(635), np.arange(507)

    coord_sys = mnt.load_mounttree(_CONFIG_PATH)
    return _ChannelGeometry(coord_sys, offset_angles, vd_vector, x_pixel, y_pixel)


def _georef_from_geometry(
    geom: _ChannelGeometry,
    lat: float, lon: float, height: float,
    roll: float, pitch: float, yaw: float,
    ref_height: float = 0.0, max_height_err: float = 0.001, flat_earth: bool = False,
) -> xr.Dataset:
    """Per-frame computation given a pre-loaded `_ChannelGeometry` -- no
    disk I/O, just coordinate-transform math. This is the part that
    genuinely differs frame to frame (aircraft position/attitude)."""
    coord_sys = geom.coord_sys
    coord_sys.update(
        lat=lat, lon=lon, height=height,
        roll=roll, pitch=pitch, yaw=yaw,
        offset_roll=geom.offset_angles['offset_roll'],
        offset_pitch=geom.offset_angles['offset_pitch'],
        offset_yaw=geom.offset_angles['offset_yaw'],
    )
    earth_frame = coord_sys.get_frame('EARTH')
    ve_transform = coord_sys.get_transformation('VELOX', 'EARTH')
    velox_position = earth_frame.toNatural(ve_transform.apply_point(0, 0, 0))
    coord_sys.update(ref_lat=velox_position[0], ref_lon=velox_position[1], ref_height=ref_height)

    vh_transform = coord_sys.get_transformation('VELOX', 'HeightPlane')
    he_transform = coord_sys.get_transformation('HeightPlane', 'EARTH')
    vd_vector = np.stack(vh_transform.apply_direction(*geom.vd_vector_velox), axis=2)

    height_error = np.inf
    height_cor = np.zeros(vd_vector[:, :, 2:3].shape)
    intersection_wgs84 = None
    while np.max(np.abs(height_error)) > max_height_err:
        scale_factor = (velox_position[2] - ref_height + height_cor) / vd_vector[:, :, 2:3]
        if np.max(scale_factor) > 1e6:
            raise ValueError('Some view direction vectors run almost parallel to the height plane!')
        if np.min(scale_factor) < 0:
            raise ValueError('Some view direction vectors never reach the height plane!')

        intersection = scale_factor * vd_vector
        intersection[:, :, 2:3] = height_cor
        intersection_earth = np.stack(he_transform.apply_point(*intersection.transpose(2, 0, 1)), axis=2)
        # toNatural() is fully vectorized (pure numpy internally) -- call it
        # once on the whole (nx, ny) grid instead of once per pixel. The
        # original PaulR script (and an earlier version of this port) did
        # `[earth_frame.toNatural(px) for px in intersection_earth.reshape(-1, 3)]`,
        # a ~327k-iteration Python loop per convergence pass -- this was the
        # actual bottleneck (~30s/frame), not the disk I/O the geometry
        # caching above addresses.
        lat_arr, lon_arr, height_arr = earth_frame.toNatural(intersection_earth.transpose(2, 0, 1))
        intersection_wgs84 = np.stack([lat_arr, lon_arr, height_arr], axis=2)

        height_error = intersection_wgs84[:, :, 2:3] - ref_height
        height_cor = height_cor + height_error

        if flat_earth:
            break

    result = xr.Dataset(
        data_vars=dict(
            lat=(('x-pixel', 'y-pixel'), intersection_wgs84[:, :, 0]),
            lon=(('x-pixel', 'y-pixel'), intersection_wgs84[:, :, 1]),
            height=(('x-pixel', 'y-pixel'), intersection_wgs84[:, :, 2]),
        ),
        coords={'x-pixel': geom.x_pixel, 'y-pixel': geom.y_pixel},
        attrs={'description': 'Georeferenced VELOX pixels (mounttree/PaulR method)', 'ref_height': ref_height},
    )
    result['lat'].attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
    result['lon'].attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
    result['height'].attrs = {'long_name': 'height above WGS84', 'units': 'meters', 'positive': 'up'}
    return result


def georef_frame(
    lat: float, lon: float, height: float,
    roll: float, pitch: float, yaw: float,
    channel: int, campaign: str = 'HALO-AC3',
    ref_height: float = 0.0, max_height_err: float = 0.001,
    flat_earth: bool = False, footprint: bool = False,
) -> xr.Dataset:
    """Georeference a single frame of one channel.

    To georeference many frames, use :func:`georef_series`: it loads the
    calibration once instead of on every call.

    Parameters
    ----------
    lat, lon : float
        Aircraft position (deg).
    height : float
        Aircraft altitude (m above WGS84).
    roll, pitch, yaw : float
        Aircraft attitude (deg); yaw is the true heading.
    channel : int
        Channel number (1, 2, 3, 5 or 6).
    campaign : {'HALO-AC3', 'PERCUSION'}, default 'HALO-AC3'
        Campaign whose boresight calibration to use.
    ref_height : float, default 0.0
        Height (m above WGS84) of the plane the pixels are projected onto:
        0 for sea level, or a terrain height.
    max_height_err : float, default 0.001
        Convergence tolerance (m) of the height-plane intersection.
    flat_earth : bool, default False
        If True, stop after the first iteration.
    footprint : bool, default False
        If True, return the 635 x 507 grid of the final ``BT_2D`` images
        (same pixel order). Otherwise the 640 x 512 grid of the calibration
        file.

    Returns
    -------
    xarray.Dataset
        ``lat``, ``lon`` and ``height`` per (x-pixel, y-pixel).
    """
    geom = _load_channel_geometry(channel, campaign, footprint)
    result = _georef_from_geometry(
        geom, lat=lat, lon=lon, height=height, roll=roll, pitch=pitch, yaw=yaw,
        ref_height=ref_height, max_height_err=max_height_err, flat_earth=flat_earth,
    )
    result.attrs['channel'] = channel
    result.attrs['campaign'] = campaign
    return result


def georef_series(
    nav: xr.Dataset, channel: int, campaign: str = 'HALO-AC3',
    ref_height: float = 0.0, max_height_err: float = 0.001, flat_earth: bool = False,
    footprint: bool = False, rows=None,
) -> xr.Dataset:
    """Georeference a series of frames of one channel.

    Parameters
    ----------
    nav : xarray.Dataset
        Navigation data at the frame times, indexed by ``time``, with
        ``lat``, ``lon``, ``alt``, ``roll``, ``pitch`` and a heading named
        ``yaw``, ``heading`` or ``hdg``. Every time step is one frame.
    channel : int
        Channel number (1, 2, 3, 5 or 6).
    campaign : {'HALO-AC3', 'PERCUSION'}, default 'HALO-AC3'
        Campaign whose boresight calibration to use.
    ref_height, max_height_err, flat_earth, footprint
        See :func:`georef_frame`.
    rows : array-like of int or slice, optional
        Only georeference these rows (y), e.g. the strip that goes into a
        pushbroom image.

    Returns
    -------
    xarray.Dataset
        ``lat``, ``lon`` and ``height`` per (time, x-pixel, y-pixel).

    Notes
    -----
    The frames are processed one after the other, typically in well under
    a second each. For long series, subset `nav` or parallelise over
    chunks of time.
    """
    _require_mounttree()
    yaw_candidates = ('yaw', 'heading', 'hdg')  # campaign.load_nav uses 'hdg'
    yaw_var = next((v for v in yaw_candidates if v in nav), None)
    if yaw_var is None:
        raise ValueError(
            f"nav dataset has none of {yaw_candidates} -- pass a heading/yaw "
            f"variable under one of those names. Available: {list(nav.data_vars)}"
        )

    geom = _load_channel_geometry(channel, campaign, footprint)
    if rows is not None:
        geom = replace(geom, vd_vector_velox=geom.vd_vector_velox[:, :, rows], y_pixel=geom.y_pixel[rows])

    frames = []
    times = []
    for t in nav['time'].values:
        row = nav.sel(time=t)
        frame = _georef_from_geometry(
            geom,
            lat=float(row['lat']), lon=float(row['lon']), height=float(row['alt']),
            roll=float(row['roll']), pitch=float(row['pitch']), yaw=float(row[yaw_var]),
            ref_height=ref_height, max_height_err=max_height_err, flat_earth=flat_earth,
        )
        frames.append(frame)
        times.append(t)

    result = xr.concat(frames, dim='time')
    result = result.assign_coords(time=('time', times))
    result.attrs['channel'] = channel
    result.attrs['campaign'] = campaign
    return result

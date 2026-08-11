# velox_tools/georef_paulr.py
"""
Per-pixel VELOX georeferencing using the `mounttree` coordinate-transform
library.

Ported from Paul R.'s Velox_GeoRef_EachPx.py
(/projekt_agmwend/home_rad/PaulR/Velox_GeoRef_Px/), which is not itself
part of this package. This is offered as a second, more precise option
alongside `velox_tools.processing.project`:

- `project()` uses one shared analytic pinhole/FOV model (35.5x28.7 deg)
  for every channel.
- `georef_frame`/`georef_series` (this module) use a per-pixel
  View-Direction-Cosine calibration (`Velox-VDC.nc`) plus a per-channel
  boresight offset angle from the VELOX_Stereography calibration -- each
  of the 5 channels points in a very slightly different direction
  (offset_roll/pitch/yaw differ by up to ~0.5 deg between channels), which
  `project()` does not account for. It also iterates on a height plane
  instead of assuming flat ground at height 0.

Requires the `mounttree` package: pip install mounttree
"""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def _require_mounttree():
    if mnt is None:
        raise ImportError(
            "georef_paulr requires the 'mounttree' package (pip install mounttree)"
        ) from _MOUNTTREE_IMPORT_ERROR


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def available_channels(campaign: str = 'HALO-AC3') -> list[str]:
    """List channels with calibration data for a given campaign."""
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


def _load_channel_geometry(channel: int, campaign: str) -> _ChannelGeometry:
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
    flat_earth: bool = False,
) -> xr.Dataset:
    """
    Georeference a single VELOX frame (one timestep, one channel) to lat/lon/height.

    For georeferencing many frames from the same channel, use
    `georef_series` instead -- it loads the VDC calibration and builds the
    coordinate tree once and reuses it, rather than repeating that (slow,
    disk-bound) setup for every call the way calling this in a loop would.

    Parameters
    ----------
    lat, lon, height : aircraft position (deg, deg, m above WGS84)
    roll, pitch, yaw : aircraft attitude (deg)
    channel : VELOX channel number (1, 2, 3, 5, or 6 for HALO-AC3)
    campaign : which campaign's boresight calibration to use
    ref_height : height (m above WGS84) of the plane pixels are projected
        onto -- 0 for sea level, or a DEM value for known terrain height
    max_height_err : convergence tolerance (m) for the height-plane
        intersection iteration
    flat_earth : if True, skip the iteration (single flat-earth pass)

    Returns
    -------
    xr.Dataset with `lat`, `lon`, `height` per (x-pixel, y-pixel), matching
    the dims of that channel's VDC calibration file.
    """
    geom = _load_channel_geometry(channel, campaign)
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
) -> xr.Dataset:
    """
    Georeference a time series of VELOX frames using aircraft nav data.

    Parameters
    ----------
    nav : xr.Dataset indexed by `time`, with `lat`, `lon`, `alt`, `roll`,
        `pitch`, and a heading/yaw variable named `yaw`, `heading`, or
        `hdg` (HALO_nav.nc/BAHAMAS convention) -- same dataset already
        used by `velox_tools.processing.pushbroom`/`project`.
    channel : VELOX channel number
    campaign : calibration campaign (default 'HALO-AC3')
    ref_height, max_height_err, flat_earth : see `georef_frame`

    Returns
    -------
    xr.Dataset with lat/lon/height per (time, x-pixel, y-pixel).

    Note
    ----
    The VDC calibration and coordinate tree are loaded once (not once per
    frame), but the per-frame height-plane intersection itself still runs
    in a plain Python loop over `nav.time` -- each frame typically takes
    well under a second once geometry is cached, but for a full flight day
    consider subsetting `nav` first or parallelizing externally (e.g.
    dask.delayed over chunks of time).
    """
    _require_mounttree()
    yaw_candidates = ('yaw', 'heading', 'hdg')  # BAHAMAS/HALO_nav.nc uses 'hdg'
    yaw_var = next((v for v in yaw_candidates if v in nav), None)
    if yaw_var is None:
        raise ValueError(
            f"nav dataset has none of {yaw_candidates} -- pass a heading/yaw "
            f"variable under one of those names. Available: {list(nav.data_vars)}"
        )

    geom = _load_channel_geometry(channel, campaign)

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

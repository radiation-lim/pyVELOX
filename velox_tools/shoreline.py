"""Validate and calibrate the georeferencing with shorelines.

A land-water boundary in a VELOX image must fall onto the reference
shoreline after georeferencing. Labelled shoreline pixels from many
coastline passages (different flights, headings and coast orientations)
constrain corrections to the boresight angles and the timing of
:mod:`velox_tools.georef_paulr`. The workflow, run interactively in
``notebooks/shoreline_calibration.ipynb``:

1. :func:`find_events` finds the passages over coastlines in the BAHAMAS
   track of a flight: stretches during which a reference shoreline runs
   through the frame.
2. :func:`cache_frames` stores a few VELOX frames and the nearby reference
   shoreline of every event (reading one HALO-(AC)3 frame decompresses
   about 3 GB, so they are read once, in bulk).
3. Shoreline pixels are labelled by hand; :class:`LabelStore` keeps them,
   and a sky flag per event, in two CSV files.
4. :func:`frames_from_labels` and :func:`fit` fit the corrections;
   :func:`body_frame_residuals` splits the residuals into along- and
   across-track components.

The reference is the OpenStreetMap coastline (© OpenStreetMap
contributors, ODbL), downloaded once from osmdata.openstreetmap.de
(about 900 MB). Any other line shapefile can be used instead.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import CRS, Geod, Transformer
from scipy.optimize import least_squares
from shapely.ops import transform

from velox_tools import campaign, georef_paulr

#: Download location of the OpenStreetMap coastline, split into small pieces.
OSM_URL = 'https://osmdata.openstreetmap.de/download/coastlines-split-4326.zip'

#: Sky flags of an event, from "not looked at yet" to "not usable".
SKY = ('unrated', 'clear', 'partly cloudy', 'cloudy', 'unusable')

#: Seconds, relative to the event time, of the frames stored per event.
FRAME_OFFSETS = (-8, -4, 0, 4, 8)

#: Scale of the fit parameters (deg for the angles, s for the time offsets):
#: the optimiser works in these units.
_SCALE = {'roll': 0.1, 'pitch': 0.1, 'yaw': 0.1, 'dt': 0.1}

_GEOD = Geod(ellps='WGS84')


def default_cache():
    """Directory of downloads and cached frames.

    ``$VELOX_SHORELINE_CACHE`` or ``<temp dir>/pyvelox-coastlines`` -- the
    same directory notebook 03 downloads its shorelines to.
    """
    return Path(os.environ.get('VELOX_SHORELINE_CACHE', Path(tempfile.gettempdir()) / 'pyvelox-coastlines'))


# --------------------------------------------------------------------------
# reference shoreline

def osm_coastline(cache=None):
    """Path to the OpenStreetMap coastline shapefile, downloaded if needed.

    Parameters
    ----------
    cache : path-like, optional
        Download directory, default :func:`default_cache`.

    Returns
    -------
    pathlib.Path
        ``<cache>/coastlines-split-4326/lines.shp`` (WGS84 lon/lat lines).
    """
    cache = Path(cache or default_cache())
    shp = cache / 'coastlines-split-4326' / 'lines.shp'
    if shp.exists():
        return shp
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'coastlines-split-4326.zip'
    if not archive.exists():
        print(f'Downloading {OSM_URL} (~900 MB)')
        partial = archive.with_suffix('.zip.part')
        with urlopen(OSM_URL, timeout=120) as response, partial.open('wb') as out:
            shutil.copyfileobj(response, out)
        partial.replace(archive)
    with ZipFile(archive) as zipped:
        zipped.extractall(cache)
    return shp


def read_shoreline(bounds, path=None):
    """Reference shoreline lines within a lon/lat box.

    Parameters
    ----------
    bounds : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)``.
    path : path-like, optional
        Line or polygon shapefile with a ``.prj``; default
        :func:`osm_coastline`. Polygons contribute their boundaries. For
        the Norwegian Polar Institute's ``S100_Land_l``, only the
        coastline (``OBJTYPE == 'Kystkontur'``) is used.

    Returns
    -------
    list of shapely.LineString
        In lon/lat, not clipped to `bounds`.
    """
    import fiona
    from shapely.geometry import shape

    path = Path(path) if path is not None else osm_coastline()
    lines = []
    with fiona.open(path) as src:
        crs = CRS.from_user_input(src.crs_wkt or 'EPSG:4326')
        to_lonlat = None
        if not crs.equals(CRS.from_epsg(4326)):
            bounds = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform_bounds(*bounds)
            to_lonlat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform
        for feature in src.filter(bbox=tuple(bounds)):
            objtype = dict(feature.properties).get('OBJTYPE')
            if objtype is not None and objtype != 'Kystkontur':
                continue  # NPI S100: skip data limits, river and lake closing lines
            geometry = shape(feature.geometry)
            if geometry.geom_type in ('Polygon', 'MultiPolygon'):
                geometry = geometry.boundary
            if to_lonlat is not None:
                geometry = transform(to_lonlat, geometry)
            lines.extend(shapely.get_parts(shapely.line_merge(geometry)
                                           if geometry.geom_type == 'MultiLineString' else geometry))
    return [line for line in lines if line.geom_type == 'LineString']


#: Download location of the EGM96 geoid grid (15', 2.7 MB).
EGM96_URL = 'https://cdn.proj.org/us_nga_egm96_15.tif'


def geoid_height(lat, lon, cache=None):
    """Height of the EGM96 geoid above the WGS84 ellipsoid (m).

    Sea level for :func:`velox_tools.georef_paulr.georef_frame`'s
    ``ref_height``: +37 m at Svalbard, -48 m at Barbados (EGM2008 differs
    by 1-2 m). The grid is downloaded once to `cache` (default
    :func:`default_cache`).
    """
    lats, lons, grid = _geoid_grid(str(cache or default_cache()))
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lats, lons), grid)
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), (np.asarray(lon, float) + 180) % 360 - 180)
    return interp(np.stack([lat.ravel(), lon.ravel()], axis=-1)).reshape(lat.shape)


@lru_cache(maxsize=2)
def _geoid_grid(cache):
    import rasterio

    path = Path(cache) / Path(EGM96_URL).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.part')
        with urlopen(EGM96_URL, timeout=120) as response, partial.open('wb') as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    with rasterio.open(path) as src:
        grid = src.read(1).astype(float)
        rows, cols = np.arange(src.height), np.arange(src.width)
        lons = np.array(src.xy(np.zeros_like(cols), cols)[0])
        lats = np.array(src.xy(rows, np.zeros_like(rows))[1])
    if lats[0] > lats[-1]:
        lats, grid = lats[::-1], grid[::-1]
    # the grid runs from -180 to 179.75 deg; repeat -180 as 180 to close the gap at the date line
    return lats, np.append(lons, lons[0] + 360), np.concatenate([grid, grid[:, :1]], axis=1)


def _local(lat, lon):
    """Transformer from lon/lat to metres east/north of (lat, lon)."""
    crs = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    return Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform


# --------------------------------------------------------------------------
# events

def flights():
    """All research flights with final VELOX data.

    Returns
    -------
    list of tuple
        ``(campaign, research_flight)``, e.g. ``('HALO-AC3', 'RF03')``.
    """
    return [(name, rf) for name, c in campaign.CAMPAIGNS.items() for rf in c['flights']]


def frame_times(campaign_name, rf, channel=3):
    """Times of the VELOX frames of a flight, within the BAHAMAS record."""
    with xr.open_dataset(campaign._path(campaign_name, rf, 'velox', channel=channel)) as ds:
        times = ds['time'].values
    nav = campaign._load_nav(campaign_name, rf)
    return times[(times >= nav.time.values[0]) & (times <= nav.time.values[-1])]


def _ring_pixels(inset):
    """Pixels along the border of the frame, `inset` (fraction) from the edges."""
    x0, x1 = round(inset * 634), round((1 - inset) * 634)
    y0, y1 = round(inset * 506), round((1 - inset) * 506)
    xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
    return np.array([(x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym)])


def _pixel_geometry(geom, pixels):
    """`geom` for a list of (x, y) pixels only, as an (n, 1) grid.

    A single pixel is repeated: mounttree drops (1, 1) dimensions.
    """
    x, y = np.asarray(pixels).reshape(-1, 2).T
    if len(x) == 1:
        x, y = np.repeat(x, 2), np.repeat(y, 2)
    return replace(geom, vd_vector_velox=geom.vd_vector_velox[:, x, y][:, :, None],
                   x_pixel=np.arange(len(x)), y_pixel=np.arange(1))


def _nav_kwargs(nav, i=None):
    """georef_paulr keywords from navigation data (at time index `i`)."""
    value = (lambda v: float(nav[v])) if i is None else (lambda v: float(nav[v][i]))
    return dict(lat=value('lat'), lon=value('lon'), height=value('alt'),
                roll=value('roll'), pitch=value('pitch'), yaw=value('hdg'))


def _axial_mean(geometry, to_local):
    """Length-weighted mean direction (math angle, deg) and straightness (0-1) of lines."""
    c = s = total = 0.0
    for part in shapely.get_parts(geometry):
        if part.geom_type != 'LineString':
            continue
        x, y = to_local(*np.asarray(part.coords).T[:2])
        dx, dy = np.diff(x), np.diff(y)
        w = np.hypot(dx, dy)
        angle = 2 * np.arctan2(dy, dx)
        c, s, total = c + (w * np.cos(angle)).sum(), s + (w * np.sin(angle)).sum(), total + w.sum()
    if total == 0:
        return np.nan, np.nan
    return np.degrees(np.arctan2(s, c) / 2), np.hypot(c, s) / total


def find_events(campaign_name, rf, channel=3, step=4, min_coast=1.0, max_duration=60.0, shoreline=None):
    """Passages of a flight over coastlines.

    Every `step`-th frame is outlined with the georeferenced border pixels
    of the frame and of its inner 70 % (15 % inset). Frames whose inner
    part contains at least `min_coast` km of reference shoreline form
    events; consecutive frames make one event, split into parts of at most
    `max_duration` s. The frame of an event is the one with the most
    shoreline in view.

    Parameters
    ----------
    campaign_name, rf : str
        Flight, e.g. ``'HALO-AC3', 'RF13'`` (see :func:`flights`).
    channel : int, default 3
        Channel whose frame times and geometry are used.
    step : int, default 4
        Check every `step`-th frame.
    min_coast : float, default 1.0
        Minimum shoreline length (km) in the inner frame.
    max_duration : float, default 60
        Longest event (s); longer stretches are split.
    shoreline : path-like, optional
        Reference shoreline for :func:`read_shoreline`.

    Returns
    -------
    pandas.DataFrame
        One row per event: ``event_id``, ``campaign``, ``flight``, ``time``
        (of the event frame), ``start``/``end``, aircraft ``lat``, ``lon``,
        ``alt``, ``gs``, ``hdg``, ``roll``, ``pitch`` at that time, the
        shoreline length ``coast_km`` in the inner frame, the angle
        ``crossing_angle`` between track and coast (90: perpendicular),
        and the ``straightness`` of the coast (1: straight line; low for
        islands and bays, which constrain both directions).
    """
    times = frame_times(campaign_name, rf, channel)[::step]
    nav = campaign.interp_nav(campaign.load_nav(times[0]), times)
    ok = np.all([np.isfinite(nav[v].values) for v in ('lat', 'lon', 'alt', 'roll', 'pitch', 'hdg')], axis=0)
    ok &= nav['alt'].values > 1000
    times, nav = times[ok], nav.isel(time=np.flatnonzero(ok))
    if len(times) == 0:
        return _empty_events()

    geom = georef_paulr._load_channel_geometry(channel, campaign_name, footprint=True)
    rings = _pixel_geometry(geom, np.concatenate([_ring_pixels(0), _ring_pixels(0.15)]))
    lonlat = np.empty((len(times), 16, 2))
    for i in range(len(times)):
        r = georef_paulr._georef_from_geometry(rings, **_nav_kwargs(nav, i), flat_earth=True)
        lonlat[i] = np.stack([r['lon'].values[:, 0], r['lat'].values[:, 0]], axis=-1)
    outer, inner = shapely.polygons(lonlat[:, :8]), shapely.polygons(lonlat[:, 8:])

    lon_min, lat_min = lonlat.reshape(-1, 2).min(0)
    lon_max, lat_max = lonlat.reshape(-1, 2).max(0)
    lines = np.array(read_shoreline((lon_min, lat_min, lon_max, lat_max), shoreline), dtype=object)
    if not len(lines):
        return _empty_events()
    tree = shapely.STRtree(lines)
    hit_frame, hit_line = tree.query(inner, predicate='intersects')

    coast_km = np.zeros(len(times))
    direction = np.full(len(times), np.nan)
    straightness = np.full(len(times), np.nan)
    for i in np.unique(hit_frame):
        inside = shapely.intersection(shapely.union_all(lines[hit_line[hit_frame == i]]), inner[i])
        coast_km[i] = _GEOD.geometry_length(inside) / 1e3
        if coast_km[i] >= min_coast:
            direction[i], straightness[i] = _axial_mean(inside, _local(float(nav['lat'][i]), float(nav['lon'][i])))

    events = []
    selected = np.flatnonzero(coast_km >= min_coast)
    if len(selected) == 0:
        return _empty_events()
    seconds = (times - times[0]) / np.timedelta64(1, 's')
    runs = np.split(selected, np.flatnonzero(np.diff(seconds[selected]) > 3 * step) + 1)
    for run in runs:
        duration = seconds[run[-1]] - seconds[run[0]]
        for part in np.array_split(run, max(1, int(np.ceil(duration / max_duration)))):
            i = part[np.argmax(coast_km[part])]
            track = 90 - float(nav['hdg'][i])  # math angle of the flight direction
            crossing = abs((direction[i] - track + 90) % 180 - 90)
            t = pd.Timestamp(times[i])
            events.append(dict(
                event_id=f'{campaign_name}_{rf}_{t:%H%M%S}', campaign=campaign_name, flight=rf,
                time=t, start=pd.Timestamp(times[part[0]]), end=pd.Timestamp(times[part[-1]]),
                lat=float(nav['lat'][i]), lon=float(nav['lon'][i]), alt=float(nav['alt'][i]),
                gs=float(nav['gs'][i]), hdg=float(nav['hdg'][i]), roll=float(nav['roll'][i]),
                pitch=float(nav['pitch'][i]), coast_km=coast_km[i], crossing_angle=crossing,
                straightness=straightness[i],
                footprint=shapely.to_wkt(outer[i], rounding_precision=6),
            ))
    return pd.DataFrame(events)


def _empty_events():
    return pd.DataFrame(columns=['event_id', 'campaign', 'flight', 'time', 'start', 'end', 'lat', 'lon', 'alt',
                                 'gs', 'hdg', 'roll', 'pitch', 'coast_km', 'crossing_angle', 'straightness',
                                 'footprint'])


def find_all_events(flight_list=None, cache=None, workers=8, **kwargs):
    """:func:`find_events` for many flights, cached as ``<cache>/events.csv``.

    Parameters
    ----------
    flight_list : list of tuple, optional
        ``(campaign, research_flight)`` pairs; default :func:`flights`.
    cache : path-like, optional
        Default :func:`default_cache`. Flights already in the file are not
        searched again; delete it to start over.
    workers : int, default 8
        Flights searched in parallel.
    **kwargs
        Passed to :func:`find_events`.

    Returns
    -------
    pandas.DataFrame
        Events of all flights, sorted by time.
    """
    path = Path(cache or default_cache()) / 'events.csv'
    done = read_events(path) if path.exists() else _empty_events()
    searched = set(done.attrs.get('searched', []))
    todo = [f for f in (flight_list or flights()) if '/'.join(f) not in searched]
    parts = [done]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(find_events, *f, **kwargs): f for f in todo}
            for future in as_completed(futures):
                f = futures[future]
                try:
                    found = future.result()
                except FileNotFoundError as e:
                    print(f'{f[0]} {f[1]}: skipped ({e})')
                    continue
                print(f'{f[0]} {f[1]}: {len(found)} events')
                parts.append(found)
                searched.add('/'.join(f))
        events = pd.concat([p for p in parts if len(p)], ignore_index=True).sort_values('time', ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as out:
            out.write('# searched: ' + ' '.join(sorted(searched)) + '\n')
            events.to_csv(out, index=False)
    return read_events(path)


def read_events(path=None):
    """The event table written by :func:`find_all_events`."""
    path = Path(path or Path(default_cache()) / 'events.csv')
    with open(path) as f:
        header = f.readline()
    events = pd.read_csv(path, comment=None, skiprows=1, parse_dates=['time', 'start', 'end'])
    events.attrs['searched'] = header.removeprefix('# searched:').split()
    return events


# --------------------------------------------------------------------------
# frame cache

def _frame_dir(cache, channel):
    return Path(cache or default_cache()) / 'frames' / f'ch{channel}'


def _cache_flight(campaign_name, rf, events, channel, offsets, cache, shoreline):
    import netCDF4

    out_dir = _frame_dir(cache, channel)
    path = campaign._path(campaign_name, rf, 'velox', channel=channel)
    with xr.open_dataset(path) as ds:
        times = ds['time'].values
    nc = netCDF4.Dataset(path)
    var = nc['BT_2D']
    if var.chunking() != 'contiguous':
        var.set_var_chunk_cache(size=4_000_000_000, nelems=1009, preemption=0.9)  # hold one time block
    transpose = var.dimensions[1] == 'y'  # PERCUSION is (time, y, x)
    written = 0
    try:
        for _, event in events.sort_values('time').iterrows():
            target = np.datetime64(event['time']) + (np.array(offsets) * 1e3).astype('timedelta64[ms]')
            index = np.searchsorted(times, target).clip(1, len(times) - 1)
            index = np.where(np.abs(times[index - 1] - target) < np.abs(times[index] - target), index - 1, index)
            keep = np.abs(times[index] - target) <= np.timedelta64(600, 'ms')
            index = np.unique(index[keep])
            frames = []
            for i in index:
                frame = np.ma.filled(var[int(i)].astype('float32'), np.nan)
                frames.append(frame.T if transpose else frame)
            ds = xr.Dataset(
                {'BT_2D': (('time', 'x', 'y'), np.stack(frames))},
                coords={'time': times[index], 'x': np.arange(635), 'y': np.arange(507)},
                attrs=dict(event_id=event['event_id'], campaign=campaign_name, research_flight=rf, channel=channel,
                           source=os.path.basename(path)),
            )
            ds.to_netcdf(out_dir / f"{event['event_id']}.nc",
                         encoding={'BT_2D': dict(zlib=True, complevel=1, least_significant_digit=2)})
            footprint = shapely.from_wkt(event['footprint'])
            lines = read_shoreline(footprint.buffer(0.05).bounds, shoreline)
            clipped = shapely.intersection(shapely.union_all(lines), footprint.buffer(0.05))
            (out_dir / f"{event['event_id']}.wkb").write_bytes(shapely.to_wkb(clipped))
            written += 1
    finally:
        nc.close()
    return written


def cache_frames(events, channel=3, offsets=FRAME_OFFSETS, cache=None, workers=6, shoreline=None):
    """Store the frames and the reference shoreline of every event.

    Writes ``<cache>/frames/ch<channel>/<event_id>.nc`` (the frames closest
    to the event time plus `offsets`, where available within 0.6 s) and
    ``<event_id>.wkb`` (the shoreline within the frame plus about 5 km).
    Events already cached are skipped.

    Parameters
    ----------
    events : pandas.DataFrame
        From :func:`find_events` or :func:`find_all_events`.
    channel : int, default 3
        Channel to store.
    offsets : sequence of float, default :data:`FRAME_OFFSETS`
        Frame times (s) relative to the event time.
    cache : path-like, optional
        Default :func:`default_cache`.
    workers : int, default 6
        Flights processed in parallel; each holds up to 4 GB of HDF5 chunk
        cache.
    shoreline : path-like, optional
        Reference shoreline for :func:`read_shoreline`.
    """
    out_dir = _frame_dir(cache, channel)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = events[[not (out_dir / f'{e}.wkb').exists() for e in events['event_id']]]
    if todo.empty:
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_cache_flight, c, rf, group, channel, offsets, cache, shoreline): (c, rf)
                   for (c, rf), group in todo.groupby(['campaign', 'flight'])}
        for future in as_completed(futures):
            c, rf = futures[future]
            try:
                print(f'{c} {rf}: {future.result()} events cached')
            except Exception as e:  # noqa: BLE001 - report and carry on with the other flights
                print(f'{c} {rf}: failed ({type(e).__name__}: {e})')


def add_frames(event_id, times, channel=3, cache=None):
    """Add the frames closest to `times` to a cached event.

    For labelling a frame between the stored ones. HALO-(AC)3 frames take
    about half a minute each to read (see :func:`cache_frames`).
    """
    frames, _ = load_event(event_id, channel, cache)
    times = pd.to_datetime(np.atleast_1d(times))
    new = campaign.load_velox(slice(times.min() - pd.Timedelta(1, 's'), times.max() + pd.Timedelta(1, 's')),
                              channels=channel).isel(band=0)
    new = new['BT_2D'].sel(time=times.values, method='nearest').drop_vars('band').load()
    merged = xr.concat([frames['BT_2D'], new], dim='time')
    merged = merged.isel(time=np.unique(merged['time'].values, return_index=True)[1])
    out = frames.drop_vars('BT_2D').drop_dims('time').assign(BT_2D=merged)
    out.to_netcdf(_frame_dir(cache, channel) / f'{event_id}.nc',
                  encoding={'BT_2D': dict(zlib=True, complevel=1, least_significant_digit=2)})


def load_event(event_id, channel=3, cache=None):
    """Cached frames and reference shoreline of an event.

    Returns
    -------
    frames : xarray.Dataset
        ``BT_2D`` (time, x, y), as in :func:`velox_tools.campaign.load_velox`.
    shoreline : shapely geometry
        Reference shoreline near the frame, lon/lat.
    """
    directory = _frame_dir(cache, channel)
    with xr.open_dataset(directory / f'{event_id}.nc') as ds:
        frames = ds.load()
    return frames, shapely.from_wkb((directory / f'{event_id}.wkb').read_bytes())


# --------------------------------------------------------------------------
# edge suggestions

def _normals(line, distance, ds=10.0):
    """Points on `line` at `distance` and the unit normals to their left."""
    a = shapely.get_coordinates(shapely.line_interpolate_point(line, distance - ds))
    b = shapely.get_coordinates(shapely.line_interpolate_point(line, distance + ds))
    t = b - a
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    return shapely.get_coordinates(shapely.line_interpolate_point(line, distance)), \
        np.stack([-t[:, 1], t[:, 0]], axis=1) / np.where(norm > 0, norm, np.nan)


def suggest_edges(bt, east, north, coast, spacing=150.0, half_width=300.0, min_contrast=1.5, inset=0.03):
    """Candidate shoreline pixels: the strongest thermal edge near the reference.

    Stations every `spacing` m along the reference shoreline are searched
    perpendicular to it, `half_width` m to either side, for the largest
    brightness-temperature gradient. An edge is kept if the land-sea
    contrast across it exceeds `min_contrast` K, has the sign of most
    other stations in the frame, and lies away from the ends of the
    search. The search window is centred on the reference, but the edge
    is located by the image alone; `half_width` must exceed the expected
    georeferencing error. Review the suggestions: cloud edges, sea ice
    and glacier fronts can pass these checks.

    Parameters
    ----------
    bt : numpy.ndarray
        Brightness temperatures (x, y) of the frame.
    east, north : numpy.ndarray
        Position (m) of every pixel, (x, y), e.g. from
        :func:`georef_corrected` in a local projection.
    coast : shapely geometry
        Reference shoreline in the same coordinates. OpenStreetMap
        coastlines have land on their left; other references may be
        oriented either way (only the consistency of the sign matters).
    spacing, half_width : float
        Station spacing and search distance (m).
    min_contrast : float, default 1.5
        Minimum absolute land-sea brightness temperature difference (K).
    inset : float, default 0.03
        Ignore the outer fraction of the frame.

    Returns
    -------
    list of tuple
        ``(x, y)`` pixels.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    nx, ny = bt.shape
    tree = cKDTree(np.column_stack([east.ravel(), north.ravel()]))
    pixel = np.hypot(east[1, ny // 2] - east[0, ny // 2], north[1, ny // 2] - north[0, ny // 2])
    ix, iy = int(inset * nx), int(inset * ny)
    border = np.concatenate([np.column_stack([east[ix:nx - ix, iy], north[ix:nx - ix, iy]]),
                             np.column_stack([east[nx - ix - 1, iy:ny - iy], north[nx - ix - 1, iy:ny - iy]]),
                             np.column_stack([east[nx - ix - 1:ix - 1:-1, ny - iy - 1],
                                              north[nx - ix - 1:ix - 1:-1, ny - iy - 1]]),
                             np.column_stack([east[ix, ny - iy - 1:iy - 1:-1], north[ix, ny - iy - 1:iy - 1:-1]])])
    inner = shapely.Polygon(border).buffer(-half_width)
    s = np.arange(-half_width, half_width + 1e-6, pixel / 2)
    stations, contrasts, edges = [], [], []
    for line in shapely.get_parts(coast):
        if line.geom_type not in ('LineString', 'LinearRing') or line.length < 2 * spacing:
            continue
        points, normals = _normals(line, np.arange(spacing / 2, line.length, spacing))
        for p, n in zip(points, normals):
            if not np.isfinite(n).all() or not inner.contains(shapely.Point(p)):
                continue
            track = p + s[:, None] * n
            dist, index = tree.query(track)
            profile = bt.ravel()[index]
            if (dist > pixel).any() or not np.isfinite(profile).all():
                continue
            profile = ndimage.gaussian_filter1d(profile, 2)
            k = int(np.argmax(np.abs(np.gradient(profile))))
            if not 0.1 * len(s) < k < 0.9 * len(s):
                continue
            contrast = profile[k + 4:].mean() - profile[:max(k - 4, 1)].mean()  # land (left) minus sea
            stations.append(index[k])
            contrasts.append(contrast)
    if not stations:
        return []
    contrasts = np.array(contrasts)
    sign = np.sign(np.median(contrasts))
    keep = (np.sign(contrasts) == sign) & (np.abs(contrasts) >= min_contrast)
    x, y = np.divmod(np.array(stations)[keep], ny)
    return list(dict.fromkeys(zip(x.tolist(), y.tolist())))


# --------------------------------------------------------------------------
# labels

class LabelStore:
    """Labelled shoreline pixels and sky flags, kept in two CSV files.

    ``points.csv`` has one row per shoreline pixel (``event_id``,
    ``campaign``, ``flight``, frame ``time``, ``channel``, ``x``, ``y``,
    and ``source``: ``'manual'`` or ``'suggested'`` by :func:`suggest_edges`);
    ``flags.csv`` the sky flag (one of :data:`SKY`) and a free-text note
    per event. Every change is written immediately.

    Parameters
    ----------
    directory : path-like
        Where to keep the files; created if needed.
    """

    POINT_COLUMNS = ['event_id', 'campaign', 'flight', 'time', 'channel', 'x', 'y', 'source']
    FLAG_COLUMNS = ['event_id', 'campaign', 'flight', 'time', 'sky', 'note']

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.points = self._read('points.csv', self.POINT_COLUMNS)
        self.flags = self._read('flags.csv', self.FLAG_COLUMNS).set_index('event_id')
        self.flags['note'] = self.flags['note'].fillna('').astype(str)

    def _read(self, name, columns):
        path = self.directory / name
        if not path.exists():
            return pd.DataFrame(columns=columns)
        table = pd.read_csv(path, parse_dates=['time'], keep_default_na=False, na_values=[''])
        if 'source' in columns and 'source' not in table:
            table['source'] = 'manual'
        return table[columns]

    def _write(self, table, name, index):
        path = self.directory / name
        tmp = path.with_suffix('.tmp')
        table.to_csv(tmp, index=index, date_format='%Y-%m-%dT%H:%M:%S.%f')
        tmp.replace(path)

    def save(self):
        """Write both files (done automatically after every change)."""
        self._write(self.points, 'points.csv', index=False)
        self._write(self.flags, 'flags.csv', index=True)

    def frame_points(self, event_id, time, channel):
        """(x, y) of the labelled pixels of one frame."""
        p = self.points
        sel = (p['event_id'] == event_id) & (p['time'] == pd.Timestamp(time)) & (p['channel'] == channel)
        return list(zip(p.loc[sel, 'x'].astype(int), p.loc[sel, 'y'].astype(int)))

    def add_point(self, event, time, channel, x, y, source='manual'):
        """Add pixel (x, y) of a frame; ignored if already labelled."""
        self.add_points(event, time, channel, [(x, y)], source)

    def add_points(self, event, time, channel, pixels, source='manual'):
        """Add several (x, y) pixels of a frame; already labelled ones are skipped."""
        existing = set(self.frame_points(event['event_id'], time, channel))
        new = list(dict.fromkeys((int(x), int(y)) for x, y in pixels if (int(x), int(y)) not in existing))
        if not new:
            return
        rows = pd.DataFrame([dict(event_id=event['event_id'], campaign=event['campaign'], flight=event['flight'],
                                  time=pd.Timestamp(time), channel=int(channel), x=x, y=y, source=source)
                             for x, y in new])
        self.points = rows if self.points.empty else pd.concat([self.points, rows], ignore_index=True)
        self.save()

    def remove_points(self, event_id, time, channel, pixels=None):
        """Remove the given (x, y) pixels of a frame, or all of them."""
        p = self.points
        sel = (p['event_id'] == event_id) & (p['time'] == pd.Timestamp(time)) & (p['channel'] == channel)
        if pixels is not None:
            pixels = {(int(x), int(y)) for x, y in pixels}
            sel &= np.array([(int(x), int(y)) in pixels for x, y in zip(p['x'], p['y'])], dtype=bool)
        self.points = p[~sel].reset_index(drop=True)
        self.save()

    def sky(self, event_id):
        """Sky flag of an event (``'unrated'`` if not set)."""
        return self.flags['sky'].get(event_id, 'unrated')

    def note(self, event_id):
        """Note of an event."""
        return self.flags['note'].get(event_id, '')

    def set_flag(self, event, sky=None, note=None):
        """Set the sky flag and/or the note of an event."""
        if sky is not None and sky not in SKY:
            raise ValueError(f'sky must be one of {SKY}')
        key = event['event_id']
        if key not in self.flags.index:
            self.flags.loc[key] = dict(campaign=event['campaign'], flight=event['flight'],
                                       time=pd.Timestamp(event['time']), sky='unrated', note='')
        if sky is not None:
            self.flags.loc[key, 'sky'] = sky
        if note is not None:
            self.flags.loc[key, 'note'] = note
        self.save()


def progress(events, store, min_points=5):
    """Labelling progress per flight.

    Parameters
    ----------
    events : pandas.DataFrame
        All events (:func:`find_all_events`).
    store : LabelStore
        Labels.
    min_points : int, default 5
        Pixels an event needs to count as labelled.

    Returns
    -------
    pandas.DataFrame
        Per flight: number of ``events``, of ``rated`` ones, of ``clear``
        and ``partly cloudy`` ones, of ``labelled`` usable ones and their
        ``points``; ``headings`` lists the headings (deg) of the labelled
        events.
    """
    counts = store.points.groupby('event_id').size()
    table = events[['campaign', 'flight', 'event_id', 'hdg']].copy()
    table['sky'] = [store.sky(e) for e in table['event_id']]
    table['points'] = table['event_id'].map(counts).fillna(0).astype(int)
    usable = table['sky'].isin(['clear', 'partly cloudy'])
    table['labelled'] = usable & (table['points'] >= min_points)
    return table.groupby(['campaign', 'flight'], sort=False).apply(lambda g: pd.Series(dict(
        events=len(g),
        rated=int((g['sky'] != 'unrated').sum()),
        clear=int((g['sky'] == 'clear').sum()),
        partly=int((g['sky'] == 'partly cloudy').sum()),
        labelled=int(g['labelled'].sum()),
        points=int(g.loc[g['labelled'], 'points'].sum()),
        headings=' '.join(f'{h:.0f}' for h in g.loc[g['labelled'], 'hdg']),
    )), include_groups=False)


# --------------------------------------------------------------------------
# fit

@dataclass
class Frame:
    """Labelled pixels of one frame with everything needed to georeference them."""
    event_id: str
    campaign: str
    flight: str
    time: np.datetime64
    x: np.ndarray
    y: np.ndarray
    geometry: object  # georef_paulr._ChannelGeometry of the labelled pixels
    nav: dict  # 10 Hz navigation within +-10 s: 't' (s after `time`), 'lat', 'lon', 'alt', 'gs', 'roll', 'pitch', 'hdg'
    to_local: object  # lon/lat -> metres east/north of the frame
    shoreline: object  # reference shoreline, local metres
    ref_height: float  # sea level above the ellipsoid (m)

    def nav_at(self, dt=0.0):
        """Aircraft position and attitude `dt` s after the frame time."""
        n = self.nav
        out = {k: float(np.interp(dt, n['t'], n[k])) for k in ('lat', 'lon', 'alt', 'gs', 'roll', 'pitch', 'hdg')}
        out['hdg'] %= 360
        return out

    def positions(self, correction=None):
        """Metres east/north of the labelled pixels for a correction.

        Parameters
        ----------
        correction : dict, optional
            Added to the boresight angles, ``roll``, ``pitch``, ``yaw``
            (deg), and to the frame time, ``dt`` plus ``dt[<flight>]`` (s).
        """
        c = correction or {}
        dt = c.get('dt', 0.0) + c.get(f'dt[{self.flight}]', 0.0)
        nav = self.nav_at(dt)
        r = georef_paulr._georef_from_geometry(
            _corrected(self.geometry, c), lat=nav['lat'], lon=nav['lon'], height=nav['alt'], roll=nav['roll'],
            pitch=nav['pitch'], yaw=nav['hdg'], ref_height=self.ref_height)
        n = len(self.x)
        return np.stack(self.to_local(r['lon'].values[:n, 0], r['lat'].values[:n, 0]), axis=-1)

    def distances(self, correction=None):
        """Distance (m) of the labelled pixels from the reference shoreline."""
        return shapely.distance(shapely.points(self.positions(correction)), self.shoreline)


@lru_cache(maxsize=16)
def _geometry(channel, campaign_name):
    return georef_paulr._load_channel_geometry(channel, campaign_name, footprint=True)


def _corrected(geometry, correction):
    offsets = geometry.offset_angles
    c = correction or {}
    return replace(geometry, offset_angles=dict(
        offset_roll=offsets['offset_roll'] + c.get('roll', 0.0),
        offset_pitch=offsets['offset_pitch'] + c.get('pitch', 0.0),
        offset_yaw=offsets['offset_yaw'] + c.get('yaw', 0.0)))


def georef_corrected(time, channel=3, correction=None, sea_level=True):
    """Georeference one frame with a correction from :func:`fit`.

    Parameters
    ----------
    time : datetime-like
        Frame time.
    channel : int, default 3
        Channel.
    correction : dict, optional
        :attr:`FitResult.correction`; ``dt[<flight>]`` is used for the
        flight of `time`.
    sea_level : bool, default True
        Project onto the EGM96 geoid instead of the ellipsoid.

    Returns
    -------
    xarray.Dataset
        ``lat`` and ``lon`` (x, y) on the 635 x 507 grid.
    """
    c = correction or {}
    name, rf = campaign.flight(time)
    dt = c.get('dt', 0.0) + c.get(f'dt[{rf}]', 0.0)
    t = pd.Timestamp(time) + pd.Timedelta(dt, 's')
    nav = campaign.interp_nav(campaign.load_nav(t), [np.datetime64(t)]).isel(time=0)
    ref_height = float(geoid_height(float(nav['lat']), float(nav['lon']))) if sea_level else 0.0
    r = georef_paulr._georef_from_geometry(_corrected(_geometry(channel, name), c), **_nav_kwargs(nav),
                                           ref_height=ref_height)
    return r.rename({'x-pixel': 'x', 'y-pixel': 'y'})[['lat', 'lon']]


def frames_from_labels(store, sky=('clear', 'partly cloudy'), channel=3, cache=None, sea_level=True,
                       sources=('manual', 'suggested')):
    """Prepare the labelled frames for :func:`fit`.

    Parameters
    ----------
    store : LabelStore
        Labels.
    sky : sequence of str, default ``('clear', 'partly cloudy')``
        Use events with these sky flags.
    channel : int, default 3
        Channel of the labels (and of the geometry).
    cache : path-like, optional
        Frame cache with the reference shorelines (:func:`cache_frames`).
    sea_level : bool, default True
        Project onto the EGM96 geoid (:func:`geoid_height`) instead of
        the ellipsoid.
    sources : sequence of str, default ``('manual', 'suggested')``
        Use labels of these sources.

    Returns
    -------
    list of Frame
    """
    points = store.points[(store.points['channel'] == channel) & store.points['source'].isin(sources)]
    usable = [e for e in points['event_id'].unique() if store.sky(e) in sky]
    points = points[points['event_id'].isin(usable)]
    frames = []
    for (event_id, time), group in points.groupby(['event_id', 'time']):
        name, rf = group['campaign'].iloc[0], group['flight'].iloc[0]
        t = np.datetime64(pd.Timestamp(time))
        nav = campaign.load_nav(pd.Timestamp(time)).sel(time=slice(t - np.timedelta64(10, 's'),
                                                                   t + np.timedelta64(10, 's')))
        hdg = np.degrees(np.unwrap(np.radians(nav['hdg'].values)))
        n = dict(t=(nav.time.values - t) / np.timedelta64(1, 's'), hdg=hdg,
                 **{k: nav[k].values.astype(float) for k in ('lat', 'lon', 'alt', 'gs', 'roll', 'pitch')})
        lat, lon = float(np.interp(0, n['t'], n['lat'])), float(np.interp(0, n['t'], n['lon']))
        to_local = _local(lat, lon)
        _, shoreline = load_event(event_id, channel, cache)
        x, y = group['x'].to_numpy(int), group['y'].to_numpy(int)
        frames.append(Frame(
            event_id=event_id, campaign=name, flight=rf, time=t, x=x, y=y,
            geometry=_pixel_geometry(_geometry(channel, name), np.stack([x, y], axis=1)), nav=n, to_local=to_local,
            shoreline=transform(to_local, shoreline),
            ref_height=float(geoid_height(lat, lon)) if sea_level else 0.0,
        ))
    return frames


@dataclass
class FitResult:
    """Result of :func:`fit`.

    Attributes
    ----------
    correction : dict
        Fitted parameters: ``roll``, ``pitch``, ``yaw`` (deg, added to the
        boresight offset angles of every channel), ``dt`` (s, added to the
        frame times) and ``dt[<flight>]``.
    std : dict
        Their standard errors from the Jacobian. Pixels of a frame are not
        independent -- :func:`jackknife` gives more honest errors.
    before, after : pandas.DataFrame
        Distance of every labelled pixel from the shoreline (m).
    """
    correction: dict
    std: dict
    before: pd.DataFrame
    after: pd.DataFrame
    params: list = field(default_factory=list)
    corr: pd.DataFrame = None

    def summary(self):
        """Median, RMS and 95th percentile of the distances, before and after."""
        rows = {}
        for name, table in [('before', self.before), ('after', self.after)]:
            d = table['distance']
            rows[name] = dict(pixels=len(d), frames=table['time'].nunique(), median=d.median(),
                              rms=np.sqrt((d ** 2).mean()), p95=d.quantile(0.95))
        return pd.DataFrame(rows).T

    def offset_angles(self, campaign_name):
        """New ``Offset-Angles`` of every channel for ``velox_mounttree.yaml``."""
        config = georef_paulr._load_config()['campaign'][campaign_name]
        return {channel: {k: round(float(v + self.correction.get(k.removeprefix('offset_'), 0.0)), 3)
                          for k, v in entry['Offset-Angles'].items()}
                for channel, entry in config.items()}


def _residual_table(frames, correction):
    rows = []
    for f in frames:
        d = f.distances(correction)
        rows.append(pd.DataFrame(dict(event_id=f.event_id, campaign=f.campaign, flight=f.flight, time=f.time,
                                      x=f.x, y=f.y, distance=d)))
    return pd.concat(rows, ignore_index=True)


def fit(frames, params=('roll', 'pitch', 'yaw', 'dt'), per_flight_dt=False, loss='soft_l1', f_scale=20.0):
    """Fit boresight and timing corrections to labelled shoreline pixels.

    Minimises the distances of the georeferenced pixels from the
    reference shoreline (robust least squares). Fit each campaign on its
    own: they have separate boresight calibrations.

    Along-track errors come from pitch (``alt * dpitch``) or timing
    (``gs * dt``), across-track ones from roll; yaw rotates the frame.
    Crossings at different headings, altitudes and speeds, and coasts at
    different angles to the track, are needed to tell them apart.

    Parameters
    ----------
    frames : list of Frame
        From :func:`frames_from_labels`.
    params : sequence of str, default ``('roll', 'pitch', 'yaw', 'dt')``
        Parameters fitted for all frames.
    per_flight_dt : bool, default False
        Also fit a time offset per flight, ``dt[<flight>]``.
    loss : str, default ``'soft_l1'``
        Loss of :func:`scipy.optimize.least_squares`; robust losses limit
        the influence of mislabelled pixels.
    f_scale : float, default 20
        Distance (m) beyond which the robust loss sets in.

    Returns
    -------
    FitResult
    """
    names = list(params)
    if per_flight_dt:
        names += [f'dt[{f}]' for f in sorted({fr.flight for fr in frames})]
    if not names:
        raise ValueError('nothing to fit: give params or per_flight_dt=True')
    scale = np.array([_SCALE[n.split('[')[0]] for n in names])
    sizes = [len(f.x) for f in frames]
    bounds = np.cumsum([0] + sizes)

    def correction(z):
        return dict(zip(names, z * scale))

    def residuals(z, subset=None):
        c = correction(z)
        return np.concatenate([frames[i].distances(c) for i in (range(len(frames)) if subset is None else subset)])

    def jacobian(z):
        r0 = residuals(z)
        jac = np.zeros((len(r0), len(names)))
        for k, name in enumerate(names):
            zk = z.copy()
            zk[k] += 1e-2
            flight = name[3:-1] if name.startswith('dt[') else None
            for i, f in enumerate(frames):
                if flight is None or f.flight == flight:
                    jac[bounds[i]:bounds[i + 1], k] = (frames[i].distances(correction(zk))
                                                       - r0[bounds[i]:bounds[i + 1]]) / 1e-2
        return jac

    solution = least_squares(residuals, np.zeros(len(names)), jac=jacobian, loss=loss, f_scale=f_scale,
                             method='trf', x_scale=1.0)
    best = correction(solution.x)

    # standard errors from the Jacobian, with the scatter of the inliers
    r = solution.fun
    inlier = np.abs(r) < 3 * max(f_scale, np.median(np.abs(r)))
    jac = solution.jac[inlier] * 1.0
    dof = max(inlier.sum() - len(names), 1)
    try:
        cov = np.linalg.inv(jac.T @ jac) * (r[inlier] ** 2).sum() / dof
        std = dict(zip(names, np.sqrt(np.diag(cov)) * scale))
        d = np.sqrt(np.diag(cov))
        corr = pd.DataFrame(cov / np.outer(d, d), index=names, columns=names)
    except np.linalg.LinAlgError:
        std, corr = dict.fromkeys(names, np.nan), None
    return FitResult(correction=best, std=std, before=_residual_table(frames, {}),
                     after=_residual_table(frames, best), params=names, corr=corr)


def jackknife(frames, by='flight', **kwargs):
    """Leave-one-group-out fits: parameter spread and held-out distances.

    Parameters
    ----------
    frames : list of Frame
        From :func:`frames_from_labels`.
    by : {'flight', 'event_id'}, default 'flight'
        Group left out in turn.
    **kwargs
        Passed to :func:`fit` (``per_flight_dt`` is not supported).

    Returns
    -------
    fits : pandas.DataFrame
        Parameters fitted without each group, and the median / RMS
        distance of that group's pixels with those parameters.
    std : pandas.Series
        Jackknife standard errors of the parameters.
    """
    groups = sorted({getattr(f, by) for f in frames})
    rows = []
    for g in groups:
        train = [f for f in frames if getattr(f, by) != g]
        test = [f for f in frames if getattr(f, by) == g]
        result = fit(train, **kwargs)
        d = np.concatenate([f.distances(result.correction) for f in test])
        rows.append(dict(left_out=g, **result.correction, median=np.median(d), rms=np.sqrt(np.mean(d ** 2))))
    fits = pd.DataFrame(rows).set_index('left_out')
    params = [c for c in fits.columns if c not in ('median', 'rms')]
    n = len(fits)
    std = np.sqrt((n - 1) / n * ((fits[params] - fits[params].mean()) ** 2).sum())
    return fits, std


def body_frame_residuals(frames, correction=None):
    """Signed distances of the labelled pixels, with the coast normal in the aircraft frame.

    For a pure shift of the image by ``along`` (m, in flight direction)
    and ``across`` (m, to starboard), the signed distance is
    ``d = along * n_along + across * n_across``. :func:`body_frame_shift`
    fits that.

    Returns
    -------
    pandas.DataFrame
        Per pixel: ``event_id``, ``campaign``, ``flight``, ``time``, ``x``,
        ``y``, ``d`` (m), ``n_along`` and ``n_across`` (components of the
        unit normal of the shoreline), and ``alt``, ``gs`` and ``hdg`` of
        the aircraft.
    """
    rows = []
    for f in frames:
        p = f.positions(correction)
        points = shapely.points(p)
        parts = shapely.get_parts(f.shoreline)
        parts = parts[np.isin(shapely.get_type_id(parts), [1, 2])]  # LineString, LinearRing
        nearest = parts[shapely.STRtree(parts).query_nearest(points, all_matches=False)[1]]
        s = shapely.line_locate_point(nearest, points)
        a = shapely.get_coordinates(shapely.line_interpolate_point(nearest, s - 10))
        b = shapely.get_coordinates(shapely.line_interpolate_point(nearest, s + 10))
        q = shapely.get_coordinates(shapely.line_interpolate_point(nearest, s))
        t = b - a
        normal = np.stack([-t[:, 1], t[:, 0]], axis=1) / np.linalg.norm(t, axis=1, keepdims=True)
        nav = f.nav_at((correction or {}).get('dt', 0.0) + (correction or {}).get(f'dt[{f.flight}]', 0.0))
        h = np.radians(nav['hdg'])
        along, across = np.array([np.sin(h), np.cos(h)]), np.array([np.cos(h), -np.sin(h)])
        rows.append(pd.DataFrame(dict(
            event_id=f.event_id, campaign=f.campaign, flight=f.flight, time=f.time, x=f.x, y=f.y,
            d=((p - q) * normal).sum(1), n_along=normal @ along, n_across=normal @ across,
            alt=nav['alt'], gs=nav['gs'], hdg=nav['hdg'],
        )))
    return pd.concat(rows, ignore_index=True)


def body_frame_shift(residuals, by=None):
    """Least-squares shift of the image along and across track.

    Parameters
    ----------
    residuals : pandas.DataFrame
        From :func:`body_frame_residuals`.
    by : str or list of str, optional
        Fit separately per group, e.g. ``'event_id'`` or ``'flight'``.

    Returns
    -------
    pandas.DataFrame
        ``along`` and ``across`` (m; positive: the image lies ahead of /
        starboard of the truth) with standard errors, the number of
        pixels, and ``conditioning`` (smallest / largest singular value of
        the normals; near 0 when the coast has one orientation only and
        the shift is constrained in one direction only).
    """
    def solve(g):
        a = g[['n_along', 'n_across']].to_numpy()
        coef, *_ = np.linalg.lstsq(a, g['d'].to_numpy(), rcond=None)
        sv = np.linalg.svd(a, compute_uv=False)
        dof = max(len(g) - 2, 1)
        sigma2 = ((g['d'].to_numpy() - a @ coef) ** 2).sum() / dof
        try:
            err = np.sqrt(np.diag(np.linalg.inv(a.T @ a)) * sigma2)
        except np.linalg.LinAlgError:
            err = [np.nan, np.nan]
        return pd.Series(dict(along=coef[0], along_err=err[0], across=coef[1], across_err=err[1],
                              pixels=len(g), conditioning=sv[-1] / sv[0]))
    if by is None:
        return solve(residuals).to_frame().T
    return residuals.groupby(by).apply(solve, include_groups=False)

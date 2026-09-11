# velox_tools/processing.py
"""Pushbroom images and analytic georeferencing of VELOX frames.

These functions work on any dataset with ``BT_2D`` (time, x, y) or
(band, time, x, y) frames plus matching aircraft navigation data (``lat``,
``lon``, ``alt``, ``gs``, ``hdg``, ``roll``, ``pitch``, indexed by
``time``). Without explicit ``nav_data`` they use the BAHAMAS data of the
research flight from the campaign archive
(:func:`velox_tools.campaign.load_nav`).
:mod:`velox_tools.campaign` wraps them for the campaign archive.
"""
import numpy as np
import pandas as pd
import xarray as xr
from haversine import inverse_haversine, Unit
from tqdm import tqdm

from velox_tools.config import load_config
from velox_tools.utils import timing_wrapper


def _default_nav_data(time):
    """BAHAMAS nav data of the research flight at `time`."""
    from velox_tools import campaign  # campaign imports this module
    return campaign.load_nav(time)


@timing_wrapper
def pushbroom(dataset, slicing_position=250, quality_flag=None, nav_data=None,
              invert_y=False, time_correction: bool = True):
    """Build a pushbroom image from a sequence of VELOX frames.

    From every frame, a strip of rows starting at `slicing_position` is cut
    out whose width equals the along-track distance the aircraft covers
    until the next frame (see :func:`compute_pixel_per_second`). The strips
    are concatenated along the flight track.

    Parameters
    ----------
    dataset : xarray.Dataset
        Frames indexed by ``time`` with dims (time, x, y) or
        (band, time, x, y). Every variable with at least three dims is
        pushbroomed, all others are dropped.
    slicing_position : int, default 250
        First image row (y) of the strip cut out of every frame.
    quality_flag : array-like of bool, optional
        One flag per frame; strips of frames flagged False are set to NaN.
    nav_data : xarray.Dataset, optional
        Navigation data with ``pitch``, ``roll``, ``alt`` and ``gs``,
        interpolated onto the frame times. Default: the BAHAMAS data of the
        research flight (:func:`velox_tools.campaign.load_nav`).
    invert_y : bool, default False
        Unused, kept for backwards compatibility.
    time_correction : bool, default True
        When True, each output column is assigned the **ground-nadir time** —
        the moment the aircraft nadir passes over the ground point imaged by
        that pixel — computed per frame from instantaneous pitch, roll,
        altitude and ground speed via ``_build_ground_time_axis``.  This
        corrects the systematic along-track bias that arises because
        ``slicing_position`` is typically forward of the instantaneous nadir
        pixel.  The output is also sorted by time to guarantee a monotonic
        coordinate, as required by ``xarray.DataArray.interp_like``.

        Set False to reproduce the legacy behaviour: a uniform
        ``pd.date_range`` spanning ``dataset_time[0]`` to
        ``dataset_time[-1]`` (no per-frame correction, no sort).

    Returns
    -------
    xarray.Dataset
        The pushbroomed variables with dims (y, time) or (band, y, time).
        ``y`` here is the across-track pixel (the ``x`` axis of the input
        frames), ``time`` the along-track axis. Attributes are copied from
        `dataset`.

    Notes
    -----
    The runtime is printed after every call.
    """
    dataset_time = dataset.time
    if nav_data is None:
        nav_data = _default_nav_data(dataset_time.values[0])
    # interp instead of an exact sel: PERCUSION frames are not on full seconds
    xrHALO = nav_data.interp(time=dataset_time)

    pixel_per_second = compute_pixel_per_second(xrHALO)

    # only the strip that ends up in the image has to be read from disk
    strip = slice(slicing_position, slicing_position + int(pixel_per_second.max()))

    dataset_variables = list(dataset.data_vars.keys())
    list_of_arrays = {varname: [] for varname in dataset_variables if len(list(dataset[varname].dims)) > 2}
    im_shape = None

    # filter out variables that are not 3D (x, y, time)


    for varname in dataset_variables:
        dims = dataset[varname].dims
        if len(list(dims)) < 3:
            continue
        if 'band' in dims:
            ims = []
            for i in range(dataset['band'].size):
                dataset_array = dataset[varname].isel(band=i, y=strip).to_numpy()
                im = concat(dataset_array, 0, pixel_per_second, quality_flag)
                ims.append(im)
            im_shape = ims[0].shape
            dataset_tuple = (["band", "y", "time"], np.stack(ims))
        else:
            dataset_array = dataset[varname].isel(y=strip).to_numpy()

            im = concat(dataset_array, 0, pixel_per_second, quality_flag)
            im_shape = im.shape
            dataset_tuple = (["y", "time"], im)


        list_of_arrays[varname].append(dataset_tuple)

    if time_correction:
        time_coord = _build_ground_time_axis(
            frame_times=dataset_time.values,
            pitch=xrHALO['pitch'].values,
            roll=xrHALO['roll'].values,
            alt=xrHALO['alt'].values,
            gs=xrHALO['gs'].values,
            pixel_per_second=pixel_per_second,
            slicing_position=slicing_position,
        )
    else:
        time_coord = pd.date_range(
            start=dataset_time.values[0],
            end=dataset_time.values[-1],
            periods=im_shape[1] if im_shape else 0,
        )

    ds_out = xr.Dataset(
        data_vars=dict(
            BT_2D=(["y", "time"], np.zeros(im_shape) if im_shape else np.empty((0, 0)))
        ),
        coords=dict(
            time=time_coord,
            y=np.arange(im_shape[0] if im_shape else 0)
        ),
        attrs=dataset.attrs
    )

    for key, value in list_of_arrays.items():
        ds_out[key] = value[0]

    if 'band' in dataset.coords:
        ds_out = ds_out.assign_coords(band=dataset['band'].values)

    if time_correction:
        ds_out = ds_out.sortby('time')

    return ds_out


def _build_ground_time_axis(frame_times, pitch, roll, alt, gs,
                             pixel_per_second, slicing_position):
    """Build a ground-nadir time axis for the pushbroom output.

    For each VELOX frame *i* and strip pixel *k* (0-based within the strip
    extracted by ``concat``), the ground-nadir time is::

        t_ground(i, k) = t_frame[i]
                         + (slicing_position + k - y_nadir[i])
                           * pix_size_along[i] / gs[i]

    where

    * ``y_nadir[i]`` — the along-track pixel index of the instantaneous nadir
      point, computed from pitch / roll / altitude via
      ``nadir_to_center_of_frame``;
    * ``pix_size_along[i]`` — the along-track ground-sample distance in
      metres per pixel at the current altitude and pitch.

    A positive offset ``slicing_position > y_nadir`` means the strip looks
    ahead of the aircraft; the ground-nadir time is therefore *later* than
    ``t_frame`` (positive delta).

    Parameters
    ----------
    frame_times : array_like of datetime64[ns], shape (n_frames,)
    pitch, roll : 1-D float arrays, shape (n_frames,)   [degrees]
    alt         : 1-D float array,  shape (n_frames,)   [metres]
    gs          : 1-D float array,  shape (n_frames,)   [m s⁻¹]
    pixel_per_second : 1-D int array, shape (n_frames,)
    slicing_position : int

    Returns
    -------
    numpy.ndarray of datetime64[ns], length = sum(pixel_per_second)
    """
    times = []
    for i in range(len(frame_times)):
        pps = int(pixel_per_second[i])
        if pps <= 0:
            continue
        _, y_nadir_f = nadir_to_center_of_frame(
            float(pitch[i]), float(roll[i]), float(alt[i])
        )
        pix_size = pixel_to_meter(
            float(pitch[i]), float(roll[i]), float(alt[i])
        )[1] / 507.0
        gs_i = float(gs[i])
        for k in range(pps):
            delta_y   = (slicing_position + k) - y_nadir_f
            delta_ns  = int(round(delta_y * pix_size / gs_i * 1e9))
            times.append(frame_times[i] + np.timedelta64(delta_ns, 'ns'))
    return np.array(times, dtype='datetime64[ns]')




def compute_pixel_per_second(nav):
    """Number of image rows the aircraft moves on between two frames.

    Ground speed divided by the along-track pixel size (from
    :func:`pixel_to_meter`), i.e. the strip width used by
    :func:`pushbroom` for 1 Hz frames.

    Parameters
    ----------
    nav : xarray.Dataset
        Navigation data at the frame times with ``pitch``, ``roll`` (deg),
        ``alt`` (m) and ``gs`` (m/s).

    Returns
    -------
    numpy.ndarray of int32
        Rows per frame.
    """
    pixel_size_along_track = np.round(pixel_to_meter(nav['pitch'], nav['roll'], nav['alt'])[1] / 507)
    ground_speed = np.round(nav['gs'])
    return np.array(np.round(ground_speed / pixel_size_along_track, 0), dtype='int32')


def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):
    """Ground footprint of a VELOX frame for a given attitude and altitude.

    Parameters
    ----------
    pitch, roll : float or array-like
        Aircraft attitude (deg).
    height : float or array-like
        Height above ground (m).
    alpha, beta : float, default 35.5, 28.7
        Full field of view across (x) and along (y) track (deg).

    Returns
    -------
    xlen, ylen : float or array-like
        Across- and along-track extent of the frame on the ground (m).
        Divide by the number of pixels (635, 507) for the pixel size.
    """
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height

    return xlen, ylen


def nadir_to_center_of_frame(pitch, roll, height, alpha=35.5, beta=28.7):
    """Pixel that sees the point directly below the aircraft.

    Parameters
    ----------
    pitch, roll : float
        Aircraft attitude (deg).
    height : float
        Height above ground (m).
    alpha, beta : float, default 35.5, 28.7
        Full field of view across (x) and along (y) track (deg).

    Returns
    -------
    pixel_x, pixel_y : int
        Pixel indices of the nadir point; (317, 253) for level flight.
        Large roll or pitch (e.g. in turns) can put the nadir point outside
        the frame -- the result is then clipped to the frame edge, so check
        the attitude against the field of view if that matters.
    """
    pitch = np.radians(pitch)
    roll = -np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    pixel_x = height * np.tan(roll)  / (pixel_to_meter(pitch, roll, height)[0] / 635) + 317
    pixel_y = height * np.tan(pitch) / (pixel_to_meter(pitch, roll, height)[1] / 507) + 253

    pixel_x = int(np.clip(np.round(pixel_x), 0, 639))
    pixel_y = int(np.clip(np.round(pixel_y), 0, 511))
    return pixel_x, pixel_y


def project(data, nav_data=None):
    """Georeference VELOX pixels with the viewing angles of the sensor.

    Every pixel is projected from the aircraft position along its viewing
    zenith and azimuth angle (``viewing_angles`` file, see
    :mod:`velox_tools.config`, rotated by the heading) onto flat ground at
    sea level, starting from the nadir point shifted for roll and pitch.
    All channels share the same viewing angles. For per-channel
    calibration and a height-plane intersection, use
    :mod:`velox_tools.georef_paulr`.

    Parameters
    ----------
    data : xarray.Dataset
        Frames with dims (time, x, y) on the 635 x 507 grid. Modified in
        place.
    nav_data : xarray.Dataset, optional
        Navigation data with ``lat``, ``lon``, ``alt``, ``gs``, ``hdg``,
        ``roll`` and ``pitch``; the nearest time step is used for every
        frame. Default: the BAHAMAS data of the research flight
        (:func:`velox_tools.campaign.load_nav`).

    Returns
    -------
    xarray.Dataset
        `data` with ``lats``/``lons`` (time, x, y), plus the viewing
        geometry (``vza``, ``vaa``, ``vaa_corrected``, ``dists``), the
        navigation data per frame (``lat``, ``lon``, ``alt``, ``gs``,
        ``heading``, ``roll``, ``pitch``), the shifted frame centres
        (``offset_centers_lat``/``_lon``) and ``angle_flag`` (True where
        roll and pitch are both below 5 deg).
    """
    if nav_data is None:
        nav_data = _default_nav_data(data.time.values[0])

    config = load_config()
    ds_vel = xr.open_dataset(config.viewing_angles)

    data['vaa'] = ds_vel['vaa'].isel(x=slice(0, 635), y=slice(0, 507))
    data['vza'] = ds_vel['vza'].isel(x=slice(0, 635), y=slice(0, 507))


    data['vaa'] = data['vaa'].expand_dims(time=data.time)
    data['vza'] = data['vza'].expand_dims(time=data.time)
    

    shape_x, shape_y = data['x'].shape[0], data['y'].shape[0]

    nav_vars = ['lat', 'lon', 'alt', 'gs', 'hdg', 'roll', 'pitch']
    lat, lon, alt, gs, heading, roll, pitch = nav_data[nav_vars].sel(time=data.time, method='nearest').to_array().values
    data['lat'] = xr.DataArray(lat, dims=['time'])
    data['lon'] = xr.DataArray(lon, dims=['time'])
    data['alt'] = xr.DataArray(alt, dims=['time'])
    data['gs'] = xr.DataArray(gs, dims=['time'])
    data['heading'] = xr.DataArray(heading, dims=['time'])
    data['roll'] = xr.DataArray(roll, dims=['time'])
    data['pitch'] = xr.DataArray(pitch, dims=['time'])

    angle_flag = (abs(data['roll']) < 5) & (abs(data['pitch']) < 5)
    data['angle_flag'] = xr.DataArray(angle_flag, dims=['time'])

    lat = np.float32(lat)
    lon = np.float32(lon)

    vaa_corrected = ((data.vaa + data.heading)%360).values
    dists = (np.tan(np.radians(data.vza)) * data.alt).values

    offset_centers_lat = np.zeros(data.time.shape)
    offset_centers_lon = np.zeros(data.time.shape)
    vza_corrected = data.vza
    for i in range(len(data.time)):
        offset_coord = nadir_to_center_of_frame(pitch[i], roll[i], alt[i])
        
        # offset_idx_x/y are always valid indices into dists (clamped in
        # nadir_to_center_of_frame), even for large roll/pitch
        offset_idx_x, offset_idx_y = offset_coord
        nadir = (lat[i], lon[i])
        dists[i] = (np.tan(np.radians(vza_corrected[i])) * alt[i])
        offset_dist = dists[i, offset_idx_x, offset_idx_y]
        offset_vaa = (heading[i] - (np.sign(roll[i]) * 90)) % 360
        offset_center = inverse_haversine(nadir, offset_dist, np.radians(offset_vaa), unit=Unit.METERS)
        offset_centers_lat[i] = offset_center[0]
        offset_centers_lon[i] = offset_center[1]


    data['vaa_corrected'] = xr.DataArray(vaa_corrected, dims=['time', 'x', 'y'])
    data['dists'] = xr.DataArray(dists, dims=['time', 'x', 'y'])
    data['offset_centers_lat'] = xr.DataArray(offset_centers_lat, dims=['time'])
    data['offset_centers_lon'] = xr.DataArray(offset_centers_lon, dims=['time'])

    def compute_projected_coordinates(offset_center_lat, offset_center_lon, vaa, dist):
        # Convert inputs to numpy arrays if they aren't already
        offset_center_lat = np.asarray(offset_center_lat)
        offset_center_lon = np.asarray(offset_center_lon)
        vaa = np.asarray(vaa)
        dist = np.asarray(dist)

        angle = np.radians(vaa)
        
        # Define a vectorized version of inverse_haversine
        vectorized_inverse_haversine = np.vectorize(inverse_haversine, excluded=['point', 'unit'])

        # Prepare output arrays
        lats_out = np.zeros_like(dist)
        lons_out = np.zeros_like(dist)

        # Calculate new coordinates
        for i in tqdm(range(offset_center_lat.shape[0])):
            offset_center = (offset_center_lat[i], offset_center_lon[i])
            lats_out[i], lons_out[i] = vectorized_inverse_haversine(point=offset_center, distance=dist[i], direction=angle[i], unit=Unit.METERS)

        return lats_out, lons_out


    lats_lons = xr.apply_ufunc(
        compute_projected_coordinates,
        data['offset_centers_lat'],
        data['offset_centers_lon'],
        data['vaa_corrected'],
        data['dists'],
        input_core_dims=[['time'], ['time'], ['time', 'x', 'y'], ['time', 'x', 'y']],
        output_core_dims=[['time', 'x', 'y'], ['time', 'x', 'y']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32, np.float32]
    )

    #lats_lons = data.map_blocks(compute_projected_coordinates, data['offset_centers_lat'], data['offset_centers_lon'], data['vaa_corrected'], data['dists'], drop_axis=[0, 1, 2])

    lats_array, lons_array = lats_lons


    data['lons'] = xr.DataArray(lons_array, dims=['time', 'x', 'y'])
    data['lats'] = xr.DataArray(lats_array, dims=['time','x', 'y'])

    return data


def concat(dataset_array, slicing_position, pixel_per_second, quality_flag=None):
    """Cut a strip out of every frame and concatenate the strips.

    The core of :func:`pushbroom`, on plain numpy arrays.

    Parameters
    ----------
    dataset_array : numpy.ndarray
        Frames, shape (time, x, y).
    slicing_position : int
        First row (y) of the strip.
    pixel_per_second : array-like of int
        Strip width per frame (see :func:`compute_pixel_per_second`).
    quality_flag : array-like of bool, optional
        Strips of frames flagged False are set to NaN.

    Returns
    -------
    numpy.ndarray
        Shape (x, sum(pixel_per_second)).
    """
    arrays_to_concat = []
    for i in range(len(dataset_array)):
        concating_array = dataset_array[i, :, slicing_position:slicing_position + pixel_per_second[i]]
        if quality_flag is not None and not quality_flag[i]:
            concating_array = np.full_like(concating_array, np.nan)
        arrays_to_concat.append(concating_array)
    return np.concatenate(arrays_to_concat, axis=1)



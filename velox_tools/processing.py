# velox_tools/processing.py

import numpy as np
import xarray as xr
import pandas as pd
import time
from velox_tools.utils import timing_wrapper
from velox_tools.config import load_config
from haversine import inverse_haversine, Direction, Unit
from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dask.distributed import Client, LocalCluster
import dask
from dask import delayed, compute
import gc

_default_nav_cache = None


def _default_nav_data():
    """Lazily load and cache the default HALO nav dataset.

    Not loaded at import time: HALO_nav.nc is 200MB+ and opening+sorting it
    takes noticeable time on some filesystems, which used to happen just
    from `import velox_tools.processing` (it was a module-level default
    argument value, evaluated once at def time -- i.e. at import). Now only
    paid the first time `pushbroom`/`project` are actually called without
    an explicit `nav_data`. Path comes from `velox_tools.config.load_config`
    (`config.yaml` if present, else a path relative to this package).
    """
    global _default_nav_cache
    if _default_nav_cache is None:
        config = load_config()
        _default_nav_cache = xr.open_dataset(config.nav_data).sortby('time')
    return _default_nav_cache


@timing_wrapper
def pushbroom(dataset, slicing_position=250, quality_flag=None, nav_data=None,
              invert_y=False, time_correction: bool = True):
    """Process a dataset and create a pushbroom image.

    Parameters
    ----------
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
    """
    dataset_time = dataset.time
    if nav_data is None:
        nav_data = _default_nav_data()
    xrHALO = nav_data.sel(time=dataset_time).interp_like(dataset_time)

    pixel_size_along_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[1] / 507)
    pixel_size_across_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[0] / 635)
    ground_speed = np.round(xrHALO['gs'])
    pixel_per_second = np.array(np.round(ground_speed / pixel_size_along_track, 0), dtype='int32')

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
                dataset_array = dataset[varname].isel(band=i).to_numpy()
                im = concat(dataset_array, slicing_position, pixel_per_second, quality_flag)
                ims.append(im)
            im_shape = ims[0].shape
            dataset_tuple = (["band", "y", "time"], np.stack(ims))
        else:
            dataset_array = dataset[varname].to_numpy()

            im = concat(dataset_array, slicing_position, pixel_per_second, quality_flag)
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




def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):
    """Convert pixel measurements to meters based on vehicle attitude."""
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height

    return xlen, ylen


def nadir_to_center_of_frame(pitch, roll, height, alpha=35.5, beta=28.7):
    """Pixel index of the instantaneous nadir point in a 640x512 frame.

    Clamps to the array bounds: large roll/pitch (steep turns, etc.) can
    put the geometric nadir point outside the frame, in which case the
    nearest edge pixel is returned rather than an out-of-range index. Do
    not treat a clamped result as literally "nadir is at this pixel" --
    check the unclamped angle against the FOV yourself if that distinction
    matters for your use case.
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
    """Georeference VELOX pixels with a shared analytic pinhole/FOV model.

    See `velox_tools.georef_paulr` for a second, more precise option that
    accounts for each channel's own boresight offset angle and iterates on
    a height plane instead of assuming flat ground at height 0.
    """
    if nav_data is None:
        nav_data = _default_nav_data()

    config = load_config()
    ds_vel = xr.open_dataset(config.viewing_angles)

    data['vaa'] = ds_vel['vaa'].isel(x=slice(0, 635), y=slice(0, 507))
    data['vza'] = ds_vel['vza'].isel(x=slice(0, 635), y=slice(0, 507))


    data['vaa'] = data['vaa'].expand_dims(time=data.time)
    data['vza'] = data['vza'].expand_dims(time=data.time)
    

    shape_x, shape_y = data['x'].shape[0], data['y'].shape[0]

    lat, lon, alt, gs, heading, roll, pitch = nav_data.sel(time=data.time, method='nearest').to_array().values
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
    """Perform a pseudo-pushbroom operation on a dataset array. In dependence of the airplane speed, the array is
    sliced and concatenated to form a pushbroom image, where each push corresponds to a second of data."""
    arrays_to_concat = []
    for i in range(len(dataset_array)):
        concating_array = dataset_array[i, :, slicing_position:slicing_position + pixel_per_second[i]]
        if quality_flag is not None and not quality_flag[i]:
            concating_array = np.full_like(concating_array, np.nan)
        arrays_to_concat.append(concating_array)
    return np.concatenate(arrays_to_concat, axis=1)



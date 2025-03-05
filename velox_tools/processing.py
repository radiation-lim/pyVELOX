# velox_tools/processing.py

import numpy as np
import xarray as xr
import pandas as pd
import time
from velox_tools.utils import timing_wrapper
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

xrHALO = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_gps_new.nc').sortby('time')

def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):

    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height

    return xlen, ylen


def nadir_to_center_of_frame(pitch, roll, height, alpha=35.5, beta=28.7):

    pitch = np.radians(pitch)
    roll = -np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    
    nadir = np.array((253, 317))

    pixel_x = height * np.tan(roll)  / (pixel_to_meter(pitch, roll, height)[0] / 635) + 317
    pixel_y = height * np.tan(pitch) / (pixel_to_meter(pitch, roll, height)[1] / 507) + 253

    return int(np.round(pixel_x)), int(np.round(pixel_y))


def project(data, nav_data=xrHALO):

    ds_vel = xr.open_dataset('/projekt_agmwend/data/HALO-AC3/02_Flights/HALO-AC3_20220314_HALO_RF04/VELOX/VELOX_327kveL/Processed/HALO-AC3_VELOX_BT_Filter_01_20220314_RF04_v2.0.nc')
    
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

    roll = np.ones(data.time.shape) * -0.0001

    lat = np.float32(lat) 
    lon = np.float32(lon)

    vaa_corrected = ((data.vaa + data.heading)%360).values
    dists = (np.tan(np.radians(data.vza)) * data.alt).values

    offset_centers_lat = np.zeros(data.time.shape)
    offset_centers_lon = np.zeros(data.time.shape)
    vza_corrected = data.vza
    for i in range(len(data.time)):
        offset_coord = nadir_to_center_of_frame(pitch[i], roll[i], alt[i])
        
        offset_idx_x, offset_idx_y = offset_coord
        nadir = (lat[i], lon[i])
        dists[i] = (np.tan(np.radians(vza_corrected[i])) * alt[i])
        #try:
        offset_dist = dists[i, offset_idx_x, offset_idx_y]
        #except IndexError: 
        #offset_dist = dists[i, 253, 317]
        offset_vaa = (heading[i] - (np.sign(roll[i]) * 90)) % 360
        #offset_vaa = heading[i]
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
    # lats_array = lats_array.reshape(shape_x, shape_y)
    # lons_array = lons_array.reshape(shape_x, shape_y)

    data['lons'] = xr.DataArray(lons_array, dims=['time', 'x', 'y'])
    data['lats'] = xr.DataArray(lats_array, dims=['time','x', 'y'])

    return data

def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):
    """Convert pixel measurements to meters based on vehicle attitude."""
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)

    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height

    return xlen, ylen

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

@timing_wrapper
def pushbroom(dataset, slicing_position=250, quality_flag=None):
    """Process a dataset and create a pushbroom image."""
    dataset_time = dataset.time
    # Load external dataset using the configured path if needed.
    # In a complete package, you might call config.load_config() to get the xr_HALO location.
    xrHALO = xr.open_dataset('/projekt_agmwend/home_rad/Joshua/HALO-AC3_unified_data/unified_gps_new.nc') \
               .sortby('time').sel(time=dataset_time).interp_like(dataset_time)

    pixel_size_along_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[1] / 507)
    pixel_size_across_track = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[0] / 635)
    ground_speed = np.round(xrHALO['gs'])
    pixel_per_second = np.array(np.round(ground_speed / pixel_size_along_track, 0), dtype='int32')

    dataset_variables = list(dataset.data_vars.keys())
    list_of_arrays = {varname: [] for varname in dataset_variables}
    im_shape = None

    for varname in dataset_variables:
        dims = dataset[varname].dims
        if 'band' in dims:
            ims = []
            for i in range(dataset['band'].size):
                dataset_array = dataset[varname].isel(band=i).to_numpy()
                im = concat(dataset_array, slicing_position, pixel_per_second, quality_flag)
                ims.append(im)
            dataset_tuple = (["band", "y", "time"], np.stack(ims))
        else:
            dataset_array = dataset[varname].to_numpy()
            im = concat(dataset_array, slicing_position, pixel_per_second, quality_flag)
            im_shape = im.shape
            dataset_tuple = (["y", "time"], im)
        list_of_arrays[varname].append(dataset_tuple)

    ds_out = xr.Dataset(
        data_vars=dict(
            BT_2D=(["y", "time"], np.zeros(im_shape) if im_shape else np.empty((0, 0)))
        ),
        coords=dict(
            time=pd.date_range(start=dataset_time.values[0], end=dataset_time.values[-1], periods=im_shape[1] if im_shape else 0),
            y=np.arange(im_shape[0] if im_shape else 0)
        ),
    )

    for key, value in list_of_arrays.items():
        ds_out[key] = value[0]

    return ds_out



# velox_tools/processing.py

import numpy as np
import xarray as xr
import pandas as pd
import time
from velox_tools.utils import timing_wrapper

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
    """Concatenate data slices from a pushbroom sensor."""
    arrays_to_concat = []
    for i in range(len(dataset_array)):
        concating_array = dataset_array[i, :, slicing_position:slicing_position + pixel_per_second[i]]
        if quality_flag is not None and not quality_flag[i]:
            concating_array = np.full_like(concating_array, np.nan)
        arrays_to_concat.append(concating_array)
    return np.concatenate(arrays_to_concat, axis=1)

@timing_wrapper
def concatenate_images3(dataset, slicing_position=250, time_slice=None, channel=0, variable='BT_2D', quality_flag=None, mode='numpy'):
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

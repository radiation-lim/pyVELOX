import os

import numpy as np
import pytest
import xarray as xr

from velox_tools import campaign
from velox_tools.config import load_config

needs_data = pytest.mark.skipif(not os.path.isdir(load_config().data_root),
                                reason='campaign archive (data_root) not mounted')


def test_flight_from_time():
    assert campaign.flight(slice('2022-03-20T10:00', '2022-03-20T10:30')) == ('HALO-AC3', 'RF07')
    assert campaign.flight('2024-08-25T12:00:00') == ('PERCUSION', 'RF07')
    assert campaign.flight('2024-11-12') == ('PERCUSION', 'ECVal_RF04')
    with pytest.raises(ValueError):
        campaign.flight('2023-01-01')


def test_data_root_from_env(monkeypatch):
    monkeypatch.setenv('VELOX_DATA_ROOT', 'P:/data')
    assert load_config('does-not-exist.yaml').data_root == 'P:/data'


def test_config_found_from_notebook_folder(tmp_path, monkeypatch):
    # Jupyter runs notebooks in their own folder, config.yaml sits in the repo root
    (tmp_path / 'config.yaml').write_text('data_root: Q:/data\n')
    (tmp_path / 'velox_tools' / 'notebooks').mkdir(parents=True)
    monkeypatch.chdir(tmp_path / 'velox_tools' / 'notebooks')
    config = load_config()
    assert config.data_root == 'Q:/data'
    assert config.config_file == str(tmp_path / 'config.yaml')


def test_interp_nav_heading_wraps():
    nav = xr.Dataset(dict(hdg=('time', [359.0, 1.0]), alt=('time', [0.0, 2.0])),
                     coords=dict(time=np.array(['2024-01-01T00:00:00', '2024-01-01T00:00:01'], dtype='datetime64[ns]')))
    out = campaign.interp_nav(nav, xr.DataArray(np.array(['2024-01-01T00:00:00.5'], dtype='datetime64[ns]'), dims='time'))
    assert float(out['alt']) == 1.0
    assert min(float(out['hdg']), 360 - float(out['hdg'])) < 1e-6  # 0 deg, not 180


@needs_data
def test_load_nav_convention():
    nav = campaign.load_nav('2024-08-27')  # BAHAMAS file is 20240827b in the 20240827a folder
    assert list(nav.data_vars) == ['lat', 'lon', 'alt', 'gs', 'hdg', 'roll', 'pitch']
    assert nav.indexes['time'].is_monotonic_increasing


@needs_data
@pytest.mark.parametrize('time', [
    slice('2022-03-20T10:51:00', '2022-03-20T10:51:04'),
    slice('2024-08-25T12:00:00', '2024-08-25T12:00:04'),
])
def test_load_velox_same_layout_for_both_campaigns(time):
    ds = campaign.load_velox(time)
    assert ds['BT_2D'].dims == ('band', 'time', 'x', 'y')
    assert ds['BT_2D'].shape[0] == 5 and ds['BT_2D'].shape[2:] == (635, 507)
    assert list(ds.band.values) == campaign.CHANNELS
    assert 'BT_Center' in ds


@needs_data
def test_georef_pushbroom_matches_full_frame_georef():
    from velox_tools import processing
    time = slice('2024-08-25T12:00:00', '2024-08-25T12:00:05')
    pb = campaign.pushbroom(time, georef=True, time_correction=False)
    full = campaign.georef(time, channel=1)
    pixel_per_second = processing.compute_pixel_per_second(campaign.load_nav(time).interp(time=full.time))
    assert np.allclose(processing.concat(full['lats'].values, 250, pixel_per_second), pb['lats'].values)


@needs_data
def test_processing_defaults_to_bahamas_nav():
    from velox_tools import processing
    time = slice('2024-08-25T12:00:00', '2024-08-25T12:00:05')
    ds = campaign.load_velox(time, channels=3)
    default = processing.pushbroom(ds)
    explicit = processing.pushbroom(ds, nav_data=campaign.load_nav(time))
    assert np.array_equal(default.time.values, explicit.time.values)

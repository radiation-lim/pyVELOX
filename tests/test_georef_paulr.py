import numpy as np
import pytest

from velox_tools.georef_paulr import available_channels, georef_frame, _channel_calibration


def test_available_channels_halo_ac3():
    channels = available_channels('HALO-AC3')
    assert channels == ['Channel1', 'Channel2', 'Channel3', 'Channel5', 'Channel6']


def test_unknown_campaign_raises():
    with pytest.raises(ValueError):
        available_channels('NOT-A-REAL-CAMPAIGN')


def test_unknown_channel_raises():
    with pytest.raises(ValueError):
        _channel_calibration(channel=99, campaign='HALO-AC3')


def test_georef_frame_gives_physically_sane_footprint():
    """Regression test for a known HALO-AC3 position (2022-03-20, RF07),
    checked by hand on 2026-08-11: aircraft at 78.673N, -14.333E, 11731m,
    should georeference to a footprint of a few km around that position --
    not, say, a different hemisphere or a footprint the size of a
    continent (both of which have happened during development from
    unit/sign errors elsewhere in a georef chain)."""
    frame = georef_frame(
        lat=78.67302837694045, lon=-14.33349151151274, height=11731.205078125,
        roll=0.0, pitch=-2.5, yaw=90.0,
        channel=3, flat_earth=True,
    )
    assert 'lat' in frame and 'lon' in frame and 'height' in frame
    assert frame['lat'].shape == (640, 512)

    lat_span = float(frame['lat'].max() - frame['lat'].min())
    lon_span = float(frame['lon'].max() - frame['lon'].min())
    # ~35x29 deg FOV at ~11.7km altitude -> a few km footprint => well under 1 degree
    assert 0 < lat_span < 1.0
    assert 0 < lon_span < 1.0
    # footprint should be centered near the aircraft, not wildly displaced
    assert abs(float(frame['lat'].mean()) - 78.673) < 0.5
    assert abs(float(frame['lon'].mean()) - (-14.333)) < 2.0


def test_georef_series_matches_georef_frame():
    """The cached-geometry series path must give the same answer as a
    fresh single-frame call for the same input (see project notes
    2026-08-11 -- an earlier optimization pass verified this numerically,
    this pins it down as a regression test)."""
    import xarray as xr
    from velox_tools.georef_paulr import georef_series

    kwargs = dict(
        lat=78.67302837694045, lon=-14.33349151151274, height=11731.205078125,
        roll=0.3, pitch=-2.5, yaw=95.0, channel=3, flat_earth=True,
    )
    single = georef_frame(**kwargs)

    nav = xr.Dataset(
        data_vars=dict(
            lat=('time', [kwargs['lat']]),
            lon=('time', [kwargs['lon']]),
            alt=('time', [kwargs['height']]),
            roll=('time', [kwargs['roll']]),
            pitch=('time', [kwargs['pitch']]),
            hdg=('time', [kwargs['yaw']]),
        ),
        coords=dict(time=[np.datetime64('2022-03-20T10:51:33')]),
    )
    series = georef_series(nav, channel=3, flat_earth=True)
    series_frame = series.isel(time=0)

    assert np.allclose(single['lat'].values, series_frame['lat'].values)
    assert np.allclose(single['lon'].values, series_frame['lon'].values)
    assert np.allclose(single['height'].values, series_frame['height'].values)

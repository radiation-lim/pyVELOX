import numpy as np
import pandas as pd
import pytest
import shapely

pytest.importorskip('mounttree')

from velox_tools import georef_paulr, shoreline  # noqa: E402

GEOD = shoreline._GEOD


def synthetic_frame(truth, hdg, alt, gs, flight, channel=3, campaign='HALO-AC3'):
    """A frame whose shoreline is where a ring of pixels lands with the correction `truth`."""
    lat0, lon0 = 70.0, 15.0
    t = np.arange(-10, 10.01, 0.1)
    lon, lat, _ = GEOD.fwd(np.full(t.shape, lon0), np.full(t.shape, lat0), np.full(t.shape, hdg), gs * t)
    nav = dict(t=t, lat=lat, lon=lon, alt=np.full(t.shape, alt), gs=np.full(t.shape, gs),
               roll=np.full(t.shape, 0.5), pitch=np.full(t.shape, 2.5), hdg=np.full(t.shape, hdg))
    angle = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    pixels = np.unique(np.stack([317 + 200 * np.cos(angle), 253 + 150 * np.sin(angle)], 1).round().astype(int), axis=0)
    geometry = georef_paulr._load_channel_geometry(channel, campaign, footprint=True)
    frame = shoreline.Frame(
        event_id=f'{flight}_{hdg:.0f}', campaign=campaign, flight=flight, time=np.datetime64('2022-04-04T09:00'),
        x=pixels[:, 0], y=pixels[:, 1], geometry=shoreline._pixel_geometry(geometry, pixels), nav=nav,
        to_local=shoreline._local(lat0, lon0), shoreline=None, ref_height=0.0)
    true = frame.positions(truth)
    order = np.argsort(np.arctan2(true[:, 1] - true[:, 1].mean(), true[:, 0] - true[:, 0].mean()))
    frame.shoreline = shapely.LinearRing(true[order])
    return frame


def test_fit_recovers_boresight_and_time_offset():
    truth = dict(roll=-0.3, pitch=0.25, yaw=0.1, dt=0.4)
    frames = [synthetic_frame(truth, hdg, alt, gs, flight)
              for hdg, alt, gs, flight in [(310, 9000, 210, 'RF01'), (130, 11500, 245, 'RF01'),
                                           (350, 12000, 235, 'RF02'), (80, 7000, 190, 'RF02')]]
    assert np.median(np.concatenate([f.distances() for f in frames])) > 50
    result = shoreline.fit(frames)
    for name, value in truth.items():
        assert result.correction[name] == pytest.approx(value, abs=0.01 if name != 'dt' else 0.02)
    assert result.after['distance'].max() < 2
    new = result.offset_angles('HALO-AC3')['Channel3']
    old = georef_paulr._load_config()['campaign']['HALO-AC3']['Channel3']['Offset-Angles']
    assert new['offset_pitch'] == pytest.approx(old['offset_pitch'] + result.correction['pitch'], abs=1e-3)


def test_body_frame_shift_of_a_time_offset_is_along_track():
    frames = [synthetic_frame(dict(dt=0.5), hdg, 10000, 200, 'RF01') for hdg in (0, 120, 240)]
    shift = shoreline.body_frame_shift(shoreline.body_frame_residuals(frames)).iloc[0]
    # the shoreline lies where the image lands 0.5 s later: the image is 100 m behind it
    assert shift['along'] == pytest.approx(-100, abs=5)
    assert abs(shift['across']) < 5
    assert shift['conditioning'] > 0.5


def test_label_store_round_trip(tmp_path):
    store = shoreline.LabelStore(tmp_path)
    event = dict(event_id='HALO-AC3_RF13_090320', campaign='HALO-AC3', flight='RF13',
                 time=pd.Timestamp('2022-04-04T09:03:20'))
    t = np.datetime64('2022-04-04T09:03:24')
    store.add_point(event, t, 3, 10, 20)
    store.add_point(event, t, 3, 11, 21)
    store.add_point(event, t, 3, 10, 20)  # duplicate
    store.set_flag(event, sky='clear', note='rocky coast')
    with pytest.raises(ValueError):
        store.set_flag(event, sky='sunny')

    again = shoreline.LabelStore(tmp_path)
    assert again.frame_points(event['event_id'], t, 3) == [(10, 20), (11, 21)]
    assert again.sky(event['event_id']) == 'clear' and again.note(event['event_id']) == 'rocky coast'
    again.remove_points(event['event_id'], t, 3, [(10, 20)])
    assert shoreline.LabelStore(tmp_path).frame_points(event['event_id'], t, 3) == [(11, 21)]
    assert again.sky('unknown') == 'unrated'


def test_footprint_is_the_same_for_all_channels():
    """All channels were calibrated with one set of ground control points on
    the channel-aligned final images: the same pixel of every channel must
    see the same ground point (with channel-specific windows: 50-70 m).
    Towards the corners the lens distortion of the channels differs."""
    kw = dict(lat=69.0, lon=18.5, height=10000.0, roll=0.0, pitch=2.5, yaw=45.0, footprint=True, flat_earth=True)
    ref = georef_paulr.georef_frame(channel=3, **kw)
    for channel in (1, 2, 5, 6):
        other = georef_paulr.georef_frame(channel=channel, **kw)
        _, _, dist = GEOD.inv(ref['lon'].values, ref['lat'].values, other['lon'].values, other['lat'].values)
        assert dist[317, 253] < 3  # < 0.3 px
        assert np.median(dist) < 20


def test_suggest_edges_finds_the_image_edge_not_the_reference():
    # 15 m pixels; land (left of the reference line, x < 0 when heading north) is 8 K colder;
    # the image edge lies 90 m east of the reference
    east, north = np.meshgrid(np.arange(635) * 15.0 - 4750, np.arange(507) * 15.0 - 3800, indexing='ij')
    bt = np.where(east < 90, -20.0, -12.0) + np.random.default_rng(0).normal(0, 0.3, east.shape)
    coast = shapely.LineString([(0, -3000), (0, 3000)])  # heading north: land (left) is west
    pixels = shoreline.suggest_edges(bt, east, north, coast)
    assert len(pixels) >= 20
    found = np.array([east[x, y] for x, y in pixels])
    assert np.median(found) == pytest.approx(90, abs=15)
    # too little contrast: no suggestions
    assert shoreline.suggest_edges(bt, east, north, shapely.LineString([(0, 3000), (0, -3000)]),
                                   min_contrast=100) == []


def test_frame_with_a_single_label():
    frame = synthetic_frame({}, 45, 10000, 220, 'RF01')
    single = shoreline.Frame(**{**frame.__dict__, 'x': frame.x[:1], 'y': frame.y[:1],
                                'geometry': shoreline._pixel_geometry(
                                    georef_paulr._load_channel_geometry(3, 'HALO-AC3', footprint=True),
                                    np.stack([frame.x[:1], frame.y[:1]], 1))})
    assert single.positions().shape == (1, 2)
    assert np.allclose(single.positions(), frame.positions()[:1])

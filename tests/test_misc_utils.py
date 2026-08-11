import numpy as np
import pytest

from velox_tools.utils import mask_percentile_outliers, make_cluster
from velox_tools.geometry import make_vza_map
from velox_tools.io import _date_from_path


def test_make_cluster_threaded_roundtrip():
    # processes=False avoids the __main__-guard requirement of process-based
    # workers, which pytest's collection process doesn't satisfy
    client, cluster = make_cluster(n_workers=1, memory_limit='512MB', processes=False)
    try:
        result = client.submit(lambda: sum(range(10))).result()
        assert result == 45
    finally:
        client.close()
        cluster.close()


def test_mask_percentile_outliers_removes_extremes():
    data = np.concatenate([np.full(98, 5.0), [1000.0, -1000.0]])
    masked = mask_percentile_outliers(data, lo=1, hi=99)
    assert np.isnan(masked[-1]) or np.isnan(masked[-2])
    assert np.sum(np.isfinite(masked)) < len(data)
    assert np.nanmean(masked) == pytest.approx(5.0)


def test_make_vza_map_shape_and_center():
    vza = make_vza_map(nx=64, ny=48, fov_x_deg=35.5, fov_y_deg=28.7)
    assert vza.dims == ('x', 'y')
    assert vza.shape == (64, 48)
    # center pixel should be ~0 (looking straight along the optical axis)
    center_val = float(vza.isel(x=32, y=24).values)
    assert center_val < 1.0
    # corner pixels should be near the edge of the FOV
    corner_val = float(vza.isel(x=0, y=0).values)
    assert corner_val > center_val


def test_date_from_path_finds_yyyymmdd():
    path = '/projekt_agmwend/data/HALO-AC3/02_Flights/HALO-AC3_20220320_HALO_RF07/VELOX/VELOX_327kveL/Processed/Additional/T4.txt'
    assert _date_from_path(path) == '20220320'


def test_date_from_path_raises_without_date():
    with pytest.raises(ValueError):
        _date_from_path('/no/date/here/T4.txt')

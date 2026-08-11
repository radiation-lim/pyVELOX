import numpy as np
import xarray as xr
import pytest

from velox_tools.correction import poly_detrend, build_correction_table, apply_correction


def test_poly_detrend_shape_and_dtype():
    frames = np.random.rand(4, 12, 10).astype('float32')
    resid = poly_detrend(frames, order=2)
    assert resid.shape == frames.shape
    assert resid.dtype == np.float32


def test_poly_detrend_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        poly_detrend(np.zeros((5, 5)), order=2)


def test_poly_detrend_recovers_fixed_pattern_under_varying_trends():
    """Each frame has a DIFFERENT large-scale linear trend plus the SAME
    small fixed pattern. After per-frame detrending and combining across
    frames, the fixed pattern should be recoverable -- this is the whole
    point of the method.

    Note: the injected pattern here is deliberately higher-frequency
    (speckle-like), matching the real VELOX pattern's character (grainy,
    not a smooth low-order shape -- see project notes 2026-08-11). A
    *smooth* low-frequency pattern (e.g. a single sine cycle across the
    frame) overlaps with the polynomial detrend basis itself, so some of
    it gets removed along with the real per-frame trend -- that's an
    inherent property of any detrend-based method, not specific to this
    implementation, and isn't the regime this correction targets.
    """
    nx, ny = 40, 32
    xx, yy = np.mgrid[0:nx, 0:ny]
    rng = np.random.default_rng(0)
    true_pattern = rng.normal(size=(nx, ny))
    true_pattern -= true_pattern.mean()

    n = 60
    frames = np.empty((n, nx, ny), dtype='float32')
    for t in range(n):
        trend = rng.uniform(-0.2, 0.2) * xx + rng.uniform(-0.2, 0.2) * yy
        offset = rng.uniform(-30, -5)  # different "scene temperature" per frame
        frames[t] = offset + true_pattern + trend + rng.normal(scale=0.05, size=(nx, ny))

    resid = poly_detrend(frames, order=1)
    recovered = np.median(resid, axis=0)
    recovered = recovered - recovered.mean()
    expected = true_pattern - true_pattern.mean()

    r = np.corrcoef(recovered.ravel(), expected.ravel())[0, 1]
    assert r > 0.9, f"expected high correlation with the injected fixed pattern, got r={r:.3f}"


def test_poly_detrend_preserves_nan():
    frames = np.random.rand(5, 10, 10).astype('float32')
    frames[0, 3, 3] = np.nan
    resid = poly_detrend(frames, order=2)
    assert np.isnan(resid[0, 3, 3])
    assert np.isfinite(resid[1:]).all()
    assert np.isfinite(np.delete(resid[0].ravel(), 3 * 10 + 3)).all()


def _make_synthetic_dataset(n_band=2, n_time=200, nx=30, ny=24, seed=1):
    rng = np.random.default_rng(seed)
    # speckle-like fixed pattern per band -- see note in
    # test_poly_detrend_recovers_fixed_pattern_under_varying_trends for why
    # this (not a smooth low-order shape) is the representative regime
    true_pattern = rng.normal(size=(n_band, nx, ny))
    true_pattern -= true_pattern.mean(axis=(1, 2), keepdims=True)

    scene_bt = rng.uniform(-30, -5, size=(n_band, n_time))
    bt_2d = np.empty((n_band, n_time, nx, ny), dtype='float32')
    for b in range(n_band):
        for t in range(n_time):
            bt_2d[b, t] = scene_bt[b, t] + true_pattern[b] + rng.normal(scale=0.05, size=(nx, ny))

    ds = xr.Dataset(
        data_vars=dict(
            BT_2D=(('band', 'time', 'x', 'y'), bt_2d),
            BT_Center=(('band', 'time'), scene_bt),
        ),
        coords=dict(band=[0, 1][:n_band], time=np.arange(n_time), x=np.arange(nx), y=np.arange(ny)),
    )
    return ds, true_pattern


def test_build_correction_table_shape():
    ds, _ = _make_synthetic_dataset()
    table = build_correction_table(ds, bin_width=5.0, min_frames_per_bin=5, order=1)
    assert 'pattern' in table
    assert 'n_frames' in table
    assert table['pattern'].dims == ('band', 'Tbin', 'x', 'y')
    assert table.sizes['band'] == 2
    assert table.sizes['Tbin'] > 0


def test_build_correction_table_min_frames_filters_bins():
    ds, _ = _make_synthetic_dataset(n_time=40)
    # extremely narrow bins + high min_frames_per_bin should leave nothing
    with pytest.raises(ValueError):
        build_correction_table(ds, bin_width=0.5, min_frames_per_bin=1000, order=1)


def test_apply_correction_reduces_spatial_std_on_synthetic_pattern():
    """End-to-end: build a table on synthetic data with a known injected
    fixed pattern, apply it, and check that applying the correction
    actually reduces per-frame spatial std relative to the raw data."""
    ds, true_pattern = _make_synthetic_dataset(n_time=300, seed=2)
    table = build_correction_table(ds, bin_width=5.0, min_frames_per_bin=5, order=1)

    band = 0
    bt_2d = ds['BT_2D'].isel(band=band)
    scene_bt = ds['BT_Center'].isel(band=band)
    corrected = apply_correction(bt_2d, scene_bt, table, band=band)

    raw_std = bt_2d.std(dim=['x', 'y']).mean().item()
    corr_std = corrected.std(dim=['x', 'y']).mean().item()
    assert corr_std < raw_std, f"expected correction to reduce spatial std: raw={raw_std:.4f} corr={corr_std:.4f}"

    # the recovered pattern for band 0 should correlate with the truth
    recovered = table['pattern'].isel(band=0).mean(dim='Tbin', skipna=True).values
    recovered = recovered - np.nanmean(recovered)
    expected = true_pattern[0] - true_pattern[0].mean()
    r = np.corrcoef(recovered.ravel(), expected.ravel())[0, 1]
    assert r > 0.8, f"expected recovered pattern to correlate with injected truth, got r={r:.3f}"


def test_apply_correction_unknown_band_raises():
    ds, _ = _make_synthetic_dataset()
    table = build_correction_table(ds, bin_width=5.0, min_frames_per_bin=5, order=1)
    bt_2d = ds['BT_2D'].isel(band=0)
    scene_bt = ds['BT_Center'].isel(band=0)
    with pytest.raises(Exception):
        apply_correction(bt_2d, scene_bt, table, band=99)

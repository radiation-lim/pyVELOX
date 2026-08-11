# velox_tools/correction.py
"""
Fixed-pattern (systematic uncertainty) correction for VELOX BT_2D imagery.

Method: for each frame, remove a low-order 2D polynomial surface (this is
what "detrend" means below) to strip out the frame's own large-scale scene
structure (temperature gradients, MIZ crossings, cloud edges, ...). Then
robustly combine (median) the per-frame residuals across many independent
frames, stratified into scene-brightness-temperature bins (the pattern's
amplitude is temperature-dependent -- worse in cold, narrowband channels).

This was validated on 2026-08-11 by cross-checking two independently
selected frame pools that share no methodology:
  - an automatic clear-sky-flagged archive (halo_ac3_velox_sea_ice_clear_sky.zarr)
  - hand/domain-expert-vetted marginal-ice-zone (MIZ) crossing timestamps
    from Mueller et al. (2024)'s velox_timesteps_in_miz_v2.csv
Both, once run through this same detrend+combine method, converge on the
same spatial pattern (Pearson r=0.74 for channel 3, 2022-04-04). Plain
time-averaging without per-frame detrending -- as used in
Mueller et al. (2024)'s correction_fields_v1/v2.nc and in this package's
own earlier notebooks/correct_fixed_pattern*.ipynb drafts -- lets real
scene gradients leak into the "correction" (that comparison showed a
spurious +-3.5K diagonal gradient with near-zero correlation to the
detrended version) and should be avoided.

A held-out validation sweep (channel 3, gaussian high-pass at several
scales vs. this polynomial detrend) found order=2 polynomial detrend gives
the best generalization to unseen frames (16.3% median spatial-std
reduction vs 12.8% for a gaussian sigma=40px baseline) -- it is the
default here.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def poly_detrend(block: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Remove a low-order 2D polynomial surface from each frame independently.

    Parameters
    ----------
    block : np.ndarray, shape (n_frames, nx, ny)
        Stack of 2D frames (e.g. BT_2D for one band, many timesteps).
    order : int
        Polynomial order (2 = quadratic surface). Validated as the
        best-performing detrend against gaussian high-pass at sigma=8/15/40.

    Returns
    -------
    np.ndarray, float32, same shape as `block`: per-frame residual after
    subtracting the fitted surface. NaNs in the input are preserved in the
    output (excluded from the fit, filled with the frame's median for
    fitting purposes only).
    """
    if block.ndim != 3:
        raise ValueError(f"poly_detrend expects (n_frames, nx, ny), got shape {block.shape}")
    n, nx, ny = block.shape
    xx, yy = np.mgrid[0:nx, 0:ny]
    xx_f = xx.ravel().astype('float64')
    yy_f = yy.ravel().astype('float64')

    terms = [np.ones_like(xx_f)]
    for o in range(1, order + 1):
        for k in range(o + 1):
            terms.append((xx_f ** (o - k)) * (yy_f ** k))
    design = np.stack(terms, axis=1)
    design_pinv = np.linalg.pinv(design)

    out = np.empty_like(block, dtype='float32')
    for t in range(n):
        frame = block[t]
        nanmask = ~np.isfinite(frame)
        flat = frame.ravel().astype('float64')
        if nanmask.any():
            fill_value = np.nanmedian(flat)
            flat = np.where(np.isnan(flat), fill_value, flat)
        coef = design_pinv @ flat
        surface = (design @ coef).reshape(nx, ny)
        resid = (frame - surface).astype('float32')
        if nanmask.any():
            resid[nanmask] = np.nan
        out[t] = resid
    return out


def _bt_bin_edges(scene_bt: np.ndarray, bin_width: float, lo_pct: float = 1, hi_pct: float = 99) -> np.ndarray:
    valid = np.isfinite(scene_bt)
    if not valid.any():
        raise ValueError("scene_bt has no finite values")
    lo, hi = np.nanpercentile(scene_bt[valid], [lo_pct, hi_pct])
    return np.arange(
        np.floor(lo / bin_width) * bin_width,
        np.ceil(hi / bin_width) * bin_width + bin_width,
        bin_width,
    )


def build_correction_table(
    ds: xr.Dataset,
    bands: list[int] | None = None,
    bin_width: float = 5.0,
    min_frames_per_bin: int = 30,
    max_frames_per_bin: int | None = None,
    order: int = 2,
    seed: int = 42,
) -> xr.Dataset:
    """
    Build a per-band, per-scene-temperature-bin fixed-pattern correction table.

    Parameters
    ----------
    ds : xr.Dataset
        Needs `BT_2D` (dims: band, time, x, y) and `BT_Center` (band, time)
        -- the per-frame scene-mean brightness temperature, used to sort
        frames into bins. Use a large, varied frame pool: a handful of
        hand-picked scenes is not enough for the median-combine to average
        out real scene structure. Validated with a clear-sky-flagged
        multi-flight archive (~150-3000 frames/bin) and independently with
        ~300-6500 MIZ-crossing frames/day; both converged to the same
        pattern once detrended.
    bands : list of int, optional
        Band indices to process (default: all bands present in `ds`).
    bin_width : float
        Width of scene-BT bins, same units as BT_Center (typically °C).
    min_frames_per_bin : int
        Skip bins with fewer candidate frames than this.
    max_frames_per_bin : int, optional
        Cap bins with more candidates than this (random subsample). None =
        use every candidate frame (validated: patterns converge by ~150
        frames/bin and don't change further with more, so capping around
        there is a reasonable speed/robustness tradeoff if needed).
    order : int
        Polynomial detrend order, see `poly_detrend`. Default 2 (validated
        best).
    seed : int
        RNG seed for max_frames_per_bin subsampling (reproducibility).

    Returns
    -------
    xr.Dataset with:
      - `pattern(band, Tbin, x, y)`: the correction map, subtract this from
        BT_2D to correct it (see `apply_correction`).
      - `n_frames(band, Tbin)`: how many frames went into each bin (NaN
        where a band has no data at that Tbin -- bin edges are computed
        per-band from that band's own BT_Center distribution, so bands
        won't all share the exact same Tbin grid).
    """
    if bands is None:
        bands = list(range(ds.sizes['band']))
    rng = np.random.default_rng(seed)

    per_band_patterns: list[np.ndarray] = []
    per_band_counts: list[np.ndarray] = []
    per_band_centers: list[np.ndarray] = []

    for b in bands:
        scene_bt = ds['BT_Center'].isel(band=b).values
        edges = _bt_bin_edges(scene_bt, bin_width)

        patterns, counts, centers = [], [], []
        for i in range(len(edges) - 1):
            e0, e1 = edges[i], edges[i + 1]
            idx = np.where(np.isfinite(scene_bt) & (scene_bt >= e0) & (scene_bt < e1))[0]
            if len(idx) < min_frames_per_bin:
                continue
            if max_frames_per_bin is not None and len(idx) > max_frames_per_bin:
                idx = rng.choice(idx, size=max_frames_per_bin, replace=False)
            idx = np.sort(idx)

            frames = ds['BT_2D'].isel(band=b, time=idx).values
            resid = poly_detrend(frames, order=order)
            pattern = np.nanmedian(resid, axis=0)

            patterns.append(pattern)
            counts.append(len(idx))
            centers.append((e0 + e1) / 2)

        if not patterns:
            raise ValueError(
                f"band {b}: no bins had >= {min_frames_per_bin} frames -- "
                f"check BT_Center or lower min_frames_per_bin"
            )
        per_band_patterns.append(np.stack(patterns))
        per_band_counts.append(np.array(counts, dtype='float64'))
        per_band_centers.append(np.array(centers))

    # bin centers can differ slightly per band (edges are percentile-based
    # per band) -- union them onto one Tbin axis, NaN where a band has no
    # data at that particular bin
    all_centers = sorted({round(float(c), 6) for centers in per_band_centers for c in centers})
    nx, ny = ds.sizes['x'], ds.sizes['y']
    pattern_out = np.full((len(bands), len(all_centers), nx, ny), np.nan, dtype='float32')
    counts_out = np.full((len(bands), len(all_centers)), np.nan)

    for bi, (patterns, counts, centers) in enumerate(zip(per_band_patterns, per_band_counts, per_band_centers)):
        for ci, c in enumerate(centers):
            j = all_centers.index(round(float(c), 6))
            pattern_out[bi, j] = patterns[ci]
            counts_out[bi, j] = counts[ci]

    return xr.Dataset(
        data_vars=dict(
            pattern=(('band', 'Tbin', 'x', 'y'), pattern_out),
            n_frames=(('band', 'Tbin'), counts_out),
        ),
        coords=dict(
            band=np.array(bands),
            Tbin=np.array(all_centers),
            x=ds['x'].values if 'x' in ds.coords else np.arange(nx),
            y=ds['y'].values if 'y' in ds.coords else np.arange(ny),
        ),
        attrs=dict(
            method='poly_detrend+robust_median',
            order=order,
            bin_width=bin_width,
            min_frames_per_bin=min_frames_per_bin,
            max_frames_per_bin=max_frames_per_bin if max_frames_per_bin is not None else -1,
            description='VELOX fixed-pattern correction table -- see velox_tools.correction module docstring',
        ),
    )


def apply_correction(
    bt_2d: xr.DataArray,
    scene_bt: xr.DataArray,
    correction_table: xr.Dataset,
    band: int,
) -> xr.DataArray:
    """
    Subtract the fixed-pattern correction from a BT_2D array.

    Parameters
    ----------
    bt_2d : xr.DataArray
        (time, x, y) or (x, y) for a single band/frame.
    scene_bt : xr.DataArray or float
        Scene-mean brightness temperature for the same frame(s) as `bt_2d`
        (typically that band's BT_Center) -- used to pick/interpolate the
        right temperature bin from `correction_table`.
    correction_table : xr.Dataset
        Output of `build_correction_table`.
    band : int
        Which band's pattern to apply.

    Returns
    -------
    xr.DataArray, same shape as `bt_2d`, with the temperature-interpolated
    pattern subtracted. Values outside the table's Tbin range are
    extrapolated from the nearest bins (a warning-free clip is not applied
    -- treat correction near the extremes of your data's temperature range
    with appropriate caution).
    """
    pattern = correction_table['pattern'].sel(band=band)
    pattern = pattern.dropna(dim='Tbin', how='all')
    if pattern.sizes['Tbin'] == 0:
        raise ValueError(f"correction_table has no data for band={band}")
    pattern_interp = pattern.interp(Tbin=scene_bt, kwargs=dict(fill_value='extrapolate'))
    return bt_2d - pattern_interp

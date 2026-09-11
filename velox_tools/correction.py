# velox_tools/correction.py
"""Empirical correction of the fixed-pattern noise in VELOX ``BT_2D``.

Channels 2, 3, 5 and 6 (and to a lesser extent 1) show a spatial pattern
that survives the non-uniformity correction of the camera and the
destriping of the final data. Its amplitude depends on the scene
temperature -- it is strongest for cold scenes in the narrowband channels.
The correction works in three steps:

1. Remove a low-order 2D polynomial surface from every frame
   (:func:`poly_detrend`). This strips the large-scale structure of the
   scene itself (temperature gradients, sea-ice edges, cloud edges).
2. Sort the frames into bins of scene temperature (``BT_Center``) and take
   the per-pixel median of the residuals in every bin
   (:func:`build_correction_table`).
3. Subtract the pattern, interpolated to the scene temperature of each
   frame (:func:`apply_correction`).

The detrending in step 1 is essential: plain time averages of the frames
keep real scene gradients, which then leak into the "correction". Two
independently selected frame pools (clear-sky frames over sea ice, and
marginal-ice-zone crossings) converge to the same pattern with this method
(Pearson r = 0.74, channel 3). On held-out frames, a 2nd-order polynomial
reduced the median spatial standard deviation by 16.3 %, more than a
Gaussian high-pass (12.8 % for sigma = 40 px), and is the default.

A table built from HALO-(AC)3 clear-sky frames over sea ice ships with the
package (``velox_tools/data/correction_table_v1.nc``). It covers scene
temperatures from -40 to -5 degC.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def poly_detrend(block: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a low-order 2D polynomial surface from each frame.

    Parameters
    ----------
    block : numpy.ndarray
        Stack of frames, shape (n_frames, nx, ny), e.g. ``BT_2D`` of one
        band.
    order : int, default 2
        Polynomial order (2 = quadratic surface).

    Returns
    -------
    numpy.ndarray of float32
        Same shape as `block`: the residual of every frame after subtracting
        its fitted surface. NaNs stay NaN (for the fit they are filled with
        the median of the frame).
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
    """Build a fixed-pattern correction table per band and scene temperature.

    Parameters
    ----------
    ds : xarray.Dataset
        ``BT_2D`` (band, time, x, y) and ``BT_Center`` (band, time), the
        scene temperature used to sort the frames into bins. Use a large,
        varied pool of frames -- a handful of scenes is not enough for the
        median to average out real scene structure. The shipped table used
        up to 800 frames per bin.
    bands : list of int, optional
        Positional indices of the bands to process (default: all).
    bin_width : float, default 5.0
        Width of the scene-temperature bins, in the units of ``BT_Center``.
    min_frames_per_bin : int, default 30
        Skip bins with fewer frames.
    max_frames_per_bin : int, optional
        Randomly subsample bins with more frames. None uses all of them.
        The pattern converges at about 150 frames per bin, so a cap of a
        few hundred saves time without changing the result.
    order : int, default 2
        Polynomial order of the detrending, see :func:`poly_detrend`.
    seed : int, default 42
        Seed for the subsampling.

    Returns
    -------
    xarray.Dataset
        ``pattern`` (band, Tbin, x, y): the pattern to subtract from
        ``BT_2D`` (see :func:`apply_correction`). ``n_frames`` (band, Tbin):
        the number of frames per bin. The bin edges follow the temperature
        distribution of each band, so a band can be NaN at some ``Tbin``.
        The ``band`` coordinate holds the positional band indices.
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
    """Subtract the fixed pattern from ``BT_2D`` of one band.

    Parameters
    ----------
    bt_2d : xarray.DataArray
        One band, dims (time, x, y) or (x, y).
    scene_bt : xarray.DataArray or float
        Scene temperature of the same frame(s), typically ``BT_Center`` of
        that band. The pattern is interpolated linearly between the
        temperature bins of `correction_table`.
    correction_table : xarray.Dataset
        Output of :func:`build_correction_table`.
    band : int
        Band of `correction_table` to use. For the shipped table, bands
        0-4 are channels 1, 2, 3, 5 and 6.

    Returns
    -------
    xarray.DataArray
        `bt_2d` with the pattern subtracted.

    Notes
    -----
    Outside the temperature range of the table, the pattern is
    extrapolated linearly from the outermost bins, without a warning.
    Check that your scenes lie within the table's ``Tbin`` range.
    """
    pattern = correction_table['pattern'].sel(band=band)
    pattern = pattern.dropna(dim='Tbin', how='all')
    if pattern.sizes['Tbin'] == 0:
        raise ValueError(f"correction_table has no data for band={band}")
    pattern_interp = pattern.interp(Tbin=scene_bt, kwargs=dict(fill_value='extrapolate'))
    return bt_2d - pattern_interp

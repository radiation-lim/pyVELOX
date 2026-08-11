![pyVELOX logo](logo.png)

# pyVELOX

This package is still very much in development, expect changes, bugs, and missing features.
Our vision is to create a codebase for dealing with the VELOX thermal imager. This includes reading, processing, and visualizing the data.

## Installation

To install the package, clone the repository and run the following command in the root directory:

```bash
pip install .
```

To run the processing tools, you will need to add HALO navigation file, including the 'lat', 'lon', 'alt', 'pitch', 'roll', and 'heading' (or 'hdg') data variables.
Add this file as `HALO_nav.nc` in the `data` directory.

## Usage

For usage examples, find the jupyter notebooks in the `notebooks` directory:
    - `notebooks\pushbroom.ipynb` - Example of how to create pushbroom images from the raw data
    - `notebooks\georeff.ipynb` - Example of how to georeference the raw data using `processing.project`

### Fixed-pattern correction (`velox_tools.correction`)

Channels 2, 3, 5, 6 (and to a lesser extent 1) show a temperature-dependent
spatial pattern that survives the onboard NUC and the L1 destriping step.
`velox_tools.correction` builds and applies an empirical per-pixel
correction for it:

```python
from velox_tools.correction import build_correction_table, apply_correction

# ds needs BT_2D (band, time, x, y) and BT_Center (band, time), from a
# large, varied frame pool -- see the function docstring for why
table = build_correction_table(ds, bin_width=5.0, max_frames_per_bin=800)

corrected = apply_correction(ds['BT_2D'].isel(band=2), ds['BT_Center'].isel(band=2), table, band=2)
```

A pre-built table for HALO-AC3 (all 5 bands, built from the campaign-wide
clear-sky-flagged archive) ships at `velox_tools/data/correction_table_v1.nc`.

Method notes and validation (cross-checked against two independent frame
pools, see `velox_tools/correction.py` module docstring): per-frame
polynomial detrend before combining across many scenes is what makes this
usable -- plain time-averaging (as in earlier drafts of this correction,
and in Mueller et al. (2024)'s `correction_fields_v1/v2.nc`) lets real
scene gradients leak into the "correction" and should be avoided.

### Georeferencing -- two options

- `processing.project` -- one shared analytic pinhole/FOV model (35.5x28.7
  deg) for all channels. Simple, no extra dependencies.
- `georef_paulr.georef_frame` / `georef_paulr.georef_series` -- per-pixel
  View-Direction-Cosine calibration plus a per-channel boresight offset
  angle (each of the 5 channels points in a very slightly different
  direction), iteratively intersected with a height plane. More precise;
  requires `pip install mounttree`. Ported from Paul R.'s
  `Velox_GeoRef_EachPx.py`.

```python
from velox_tools.georef_paulr import georef_series
import xarray as xr

nav = xr.open_dataset('velox_tools/data/HALO_nav.nc').sortby('time')
nav_segment = nav.sel(time=slice('2022-03-20T10:51:00', '2022-03-20T10:51:10'))
result = georef_series(nav_segment, channel=3)  # lat/lon/height per (time, x-pixel, y-pixel)
```

`georef_series` loads the calibration and coordinate geometry once and
reuses it across the whole series -- looping `georef_frame` yourself
instead will re-do that (disk-bound) setup every call.

## Known issues

- `processing.project` sets `roll = np.ones(data.time.shape) * -0.0001`
  right after reading the real roll angle from nav data, effectively
  ignoring it in part of the projection. Not fixed here (out of scope for
  this pass) -- if you rely on `project`'s absolute positioning accuracy,
  be aware of this; `georef_paulr` does not have this issue.
- `pyproject.toml`/`setup.py` paths (e.g. `HALO_nav.nc` lookups in
  `processing.py`) are hardcoded absolute paths tied to one user's home
  directory rather than resolved relative to the package -- works on this
  cluster, not portable elsewhere as-is.

## How do I get the data?

The data is not included in this repository. You can access the data [here](https://doi.pangaea.de/10.1594/PANGAEA.963401).
CAUTION: The datasets are large (~20GB) and may take a while to download.

## Processing tools for 2D thermal imagery data form the VELOX thermal infrared camera

## Documentation

API reference + the example notebooks above, built with Sphinx + MyST:

```bash
pip install -r docs/requirements.txt
cd docs && python -m sphinx -b html . _build/html
```

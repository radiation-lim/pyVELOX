# pyVELOX

Tools for processing thermal imagery from the VELOX airborne thermal
imager: reading, pushbroom compositing, georeferencing, and correcting the
temperature-dependent fixed-pattern noise that survives the onboard NUC.

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/correction_example.ipynb
notebooks/georef_paulr_example.ipynb
```

```{toctree}
:maxdepth: 2
:caption: API Reference

apidocs/velox_tools/velox_tools
```

## Quick start

```python
from velox_tools.correction import build_correction_table, apply_correction

# ds needs BT_2D (band, time, x, y) and BT_Center (band, time)
table = build_correction_table(ds, bin_width=5.0, max_frames_per_bin=800)
corrected = apply_correction(ds['BT_2D'].isel(band=2), ds['BT_Center'].isel(band=2), table, band=2)
```

```python
from velox_tools.georef_paulr import georef_series
import xarray as xr

nav = xr.open_dataset('velox_tools/data/HALO_nav.nc').sortby('time')
segment = nav.sel(time=slice('2022-03-20T10:51:00', '2022-03-20T10:51:10'))
result = georef_series(segment, channel=3)  # lat/lon/height per (time, x-pixel, y-pixel)
```

See the example notebooks above for full walkthroughs, and the API
reference for every function's parameters.

![pyVELOX logo](logo.png)

# pyVELOX

Python tools for VELOX, the airborne thermal infrared imager of the Leipzig
Institute for Meteorology on the research aircraft HALO:

- load the final brightness-temperature data of HALO-(AC)3 and PERCUSION by time,
- build pushbroom images from the 1 Hz frames,
- georeference every pixel, with a per-channel calibration of the view directions,
- correct the fixed-pattern noise of the camera.

**Documentation:** <https://radiation-lim.github.io/pyVELOX/> -- example
notebooks and the full API reference.

<!-- docs:start -->
## Installation

```bash
git clone https://github.com/radiation-lim/pyVELOX.git
cd pyVELOX
pip install .
```

For the shoreline calibration tool, install the extras with `pip install ".[shoreline]"`.

The VELOX and BAHAMAS data are read from the campaign archive on the server
(see [Configuration](#configuration)); the calibration files ship with the
package. No further downloads are needed.

## Quickstart

A time is all it takes! Campaign and research flight follow from the date:

```python
from velox_tools import campaign

campaign.flight('2024-08-25')  # ('PERCUSION', 'RF07')

# brightness temperatures of all channels: BT_2D (band, time, x, y) in degC
ds = campaign.load_velox(slice('2024-08-25T12:30:00', '2024-08-25T12:30:09'))

# pushbroom image of a flight segment, with latitude/longitude per pixel
pb = campaign.pushbroom(slice('2024-08-25T12:24', '2024-08-25T12:36'), georef=True)

# per-pixel georeferencing of full frames of one channel
geo = campaign.georef(slice('2024-08-25T12:30:00', '2024-08-25T12:30:09'), channel=3)
```

The building blocks work on any xarray dataset, e.g. to correct the fixed
pattern before making a pushbroom image:

```python
from importlib.resources import files

import xarray as xr

from velox_tools import campaign, processing
from velox_tools.correction import apply_correction

t = slice('2022-04-04T10:20', '2022-04-04T10:30')
ds = campaign.load_velox(t, channels=3).sel(band=3)

# fixed-pattern correction with the shipped table (band 2 = channel 3)
table = xr.open_dataset(files('velox_tools') / 'data' / 'correction_table_v1.nc')
ds['BT_2D'] = apply_correction(ds['BT_2D'], ds['BT_Center'], table, band=2)

pb = processing.pushbroom(ds, nav_data=campaign.load_nav(t))
```

<!-- docs:end -->
## Examples

The notebooks in [`velox_tools/notebooks`](velox_tools/notebooks) walk through
each part of the package with real data:

| Notebook | Content |
|---|---|
| [1. Loading data](velox_tools/notebooks/01_loading_data.ipynb) | configuration, research flights, brightness temperatures, navigation data |
| [2. Pushbroom images](velox_tools/notebooks/02_pushbroom_images.ipynb) | strip width, pushbroom images, ground-time axis, georeferenced pushbroom images |
| [3. Georeferencing](velox_tools/notebooks/03_georeferencing.ipynb) | single frames and series, channel offsets, comparison of both methods, cloud-top height, coastal validation |
| [4. Fixed-pattern correction](velox_tools/notebooks/04_fixed_pattern_correction.ipynb) | the shipped correction table, applying it, building your own |
| [5. Geometry and utilities](velox_tools/notebooks/05_geometry_and_utilities.ipynb) | footprint, nadir pixel, viewing angles, housekeeping temperatures, dask cluster |
| [6. Shoreline validation](velox_tools/notebooks/06_shoreline_validation.ipynb) | accuracy of the georeferencing at labelled coastlines, along- and across-track misses, roll correction of HALO-(AC)3, before and after |

[`shoreline_calibration.ipynb`](velox_tools/notebooks/shoreline_calibration.ipynb) is the interactive tool
behind notebook 6 (needs `ipympl`): it finds the coastline passages of all
flights, lets you rate and label the shoreline in their frames, and fits
boresight and timing corrections of the georeferencing to the labels.

<!-- docs:reference-start -->
## Package overview

| Module | Purpose |
|---|---|
| `velox_tools.campaign` | load VELOX and BAHAMAS data of HALO-(AC)3 and PERCUSION by time; one-call pushbroom and georeferencing |
| `velox_tools.processing` | pushbroom images and the analytic georeferencing `project` |
| `velox_tools.georef_paulr` | per-pixel georeferencing with calibrated view directions and per-channel boresight offsets |
| `velox_tools.shoreline` | validate and calibrate the georeferencing with labelled shorelines |
| `velox_tools.correction` | build and apply the fixed-pattern correction |
| `velox_tools.geometry` | viewing zenith angles of an ideal pinhole camera |
| `velox_tools.io` | lens and window temperature logs |
| `velox_tools.config` | location of the campaign archive |
| `velox_tools.utils` | timing, dask cluster, outlier masking |

### Two georeferencing methods

- `processing.project` projects each pixel along the calibrated viewing angles
  of the sensor onto flat ground at sea level. All channels share the same
  angles.
- `georef_paulr.georef_frame` / `georef_series` (the default of
  `campaign.georef`) use a per-pixel view-direction calibration of every channel
  and per-channel, per-campaign boresight offsets, and intersect the view
  directions with a plane at `ref_height` (e.g. the cloud-top height). This
  uses the [`mounttree`](https://pypi.org/project/mounttree/) package, which
  is installed with pyVELOX.

## Configuration

The data are read below `data_root`, the directory holding the `HALO-AC3/`
and `PERCUSION/` campaign archives. On the server (`/projekt_agmwend/data`) and
on Windows with `/projekt_agmwend` mounted as `P:` it is found automatically.
For any other mount, set the environment variable `VELOX_DATA_ROOT` or put a
`config.yaml` in the working directory or any parent directory:

```yaml
data_root: Q:/data
```

## Notes

- The final images are 635 x 507 pixels. The flight direction is +y, and
  x = 0 is on the starboard side.
- The HALO-(AC)3 files are chunked 2222 frames deep, so even a few seconds
  of data read about 3 GB per channel. Over a network mount, cut out one
  longer segment rather than many short ones.
- The shipped correction table covers scene temperatures from -40 to -5 °C
  (Arctic). Outside this range the pattern is extrapolated.
- `processing.project` uses the same viewing angles for all channels and
  ignores the offsets between the channel footprints (a few pixels).
- The PERCUSION files describe `vaa` as counterclockwise, the HALO-(AC)3
  files as clockwise. The values are identical; clockwise matches the images.

## References

- Schäfer, M., Wolf, K., Ehrlich, A., Hallbauer, C., Jäkel, E., Jansen, F.,
  Luebke, A. E., Müller, J., Thoböll, J., Röschenthaler, T., Stevens, B., and
  Wendisch, M. (2022): VELOX – a new thermal infrared imager for airborne
  remote sensing of cloud and surface properties, Atmos. Meas. Tech., 15,
  1491–1509, <https://doi.org/10.5194/amt-15-1491-2022>.
- Schäfer, M., Rosenburg, S., Ehrlich, A., Röttenbacher, J., and Wendisch, M.
  (2023): Two-dimensional cloud-top and surface brightness temperature with
  1 Hz temporal resolution derived at flight altitude from VELOX during the
  HALO-(AC)³ field campaign, PANGAEA, <https://doi.org/10.1594/PANGAEA.963401>.

<!-- docs:reference-end -->
## Development

```bash
pip install -e . pytest
pytest                                   # tests that need the campaign archive are skipped without it

pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

The documentation is built and published by GitHub Actions on every push to
`main`. The notebooks are rendered with their stored outputs.

## License

MIT, see [LICENSE](LICENSE).

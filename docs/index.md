# pyVELOX

Python tools for VELOX, the airborne thermal infrared imager of the Leipzig
Institute for Meteorology on the research aircraft HALO. The package loads the
final brightness-temperature data of the HALO-(AC)3 and PERCUSION campaigns,
builds pushbroom images, georeferences every pixel and corrects the fixed-pattern
noise of the camera.

## Installation

```bash
git clone https://github.com/radiation-lim/pyVELOX.git
cd pyVELOX
pip install .
```

## Quickstart

```python
from velox_tools import campaign

# brightness temperatures of all channels: BT_2D (band, time, x, y) in degC
ds = campaign.load_velox(slice('2024-08-25T12:30:00', '2024-08-25T12:30:09'))

# pushbroom image of a flight segment, with latitude/longitude per pixel
pb = campaign.pushbroom(slice('2024-08-25T12:24', '2024-08-25T12:36'), georef=True)

# per-pixel georeferencing of full frames of one channel
geo = campaign.georef(slice('2024-08-25T12:30:00', '2024-08-25T12:30:09'), channel=3)
```

The VELOX and BAHAMAS data are read from the campaign archive on the server
(see {doc}`notebooks/01_loading_data`); the calibration files ship with the
package. The building blocks in `processing`, `georef_paulr` and `correction`
take xarray datasets.

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/01_loading_data.ipynb
notebooks/02_pushbroom_images.ipynb
notebooks/03_georeferencing.ipynb
notebooks/04_fixed_pattern_correction.ipynb
notebooks/05_geometry_and_utilities.ipynb
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```

# pyVELOX

Python tools for VELOX, the airborne thermal infrared imager of the Leipzig
Institute for Meteorology on the research aircraft HALO:

- load the final brightness-temperature data of HALO-(AC)3 and PERCUSION by time,
- build pushbroom images from the 1 Hz frames,
- georeference every pixel, with a per-channel calibration of the view directions,
- correct the fixed-pattern noise of the camera.

```{include} ../README.md
:start-after: <!-- docs:start -->
:end-before: <!-- docs:end -->
```

The example notebooks walk through each part of the package with real data. The
{doc}`API reference <api/index>` documents every function.

```{toctree}
:maxdepth: 1
:caption: Examples

notebooks/01_loading_data.ipynb
notebooks/02_pushbroom_images.ipynb
notebooks/03_georeferencing.ipynb
notebooks/04_fixed_pattern_correction.ipynb
notebooks/05_geometry_and_utilities.ipynb
```

```{include} ../README.md
:start-after: <!-- docs:reference-start -->
:end-before: <!-- docs:reference-end -->
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

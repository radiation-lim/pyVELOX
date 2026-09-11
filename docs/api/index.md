# API reference

Every public function of the package, by module.

```{toctree}
:maxdepth: 1
:hidden:

campaign
processing
georef_paulr
shoreline
correction
geometry
io
config
utils
```

```{eval-rst}
.. currentmodule:: velox_tools

:doc:`campaign` -- loading campaign data by time

.. autosummary::

   campaign.flight
   campaign.load_velox
   campaign.load_nav
   campaign.interp_nav
   campaign.pushbroom
   campaign.georef

:doc:`processing` -- pushbroom images and analytic georeferencing

.. autosummary::

   processing.pushbroom
   processing.compute_pixel_per_second
   processing.concat
   processing.pixel_to_meter
   processing.nadir_to_center_of_frame
   processing.project

:doc:`georef_paulr` -- per-pixel georeferencing with calibrated view directions

.. autosummary::

   georef_paulr.available_channels
   georef_paulr.georef_frame
   georef_paulr.georef_series

:doc:`shoreline` -- validating and calibrating the georeferencing with shorelines

.. autosummary::

   shoreline.default_cache
   shoreline.osm_coastline
   shoreline.read_shoreline
   shoreline.geoid_height
   shoreline.flights
   shoreline.frame_times
   shoreline.find_events
   shoreline.find_all_events
   shoreline.read_events
   shoreline.cache_frames
   shoreline.add_frames
   shoreline.load_event
   shoreline.suggest_edges
   shoreline.LabelStore
   shoreline.progress
   shoreline.Frame
   shoreline.georef_corrected
   shoreline.frames_from_labels
   shoreline.FitResult
   shoreline.fit
   shoreline.jackknife
   shoreline.body_frame_residuals
   shoreline.body_frame_shift

:doc:`correction` -- fixed-pattern correction

.. autosummary::

   correction.poly_detrend
   correction.build_correction_table
   correction.apply_correction

:doc:`geometry` -- analytic viewing geometry

.. autosummary::

   geometry.make_vza_map

:doc:`io` -- housekeeping data

.. autosummary::

   io.load_instrument_temperatures

:doc:`config` -- paths to the input data

.. autosummary::

   config.DataConfig
   config.find_config
   config.load_config

:doc:`utils` -- utilities

.. autosummary::

   utils.timing_wrapper
   utils.make_cluster
   utils.mask_percentile_outliers
```

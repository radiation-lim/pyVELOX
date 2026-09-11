# velox_tools/geometry.py
"""Analytic viewing geometry of an ideal pinhole camera.

For quick looks. The calibrated viewing angles
(``VELOX_viewing_angles.nc``, used by
:func:`velox_tools.processing.project`) and the per-channel calibration of
:mod:`velox_tools.georef_paulr` are more accurate.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def make_vza_map(nx: int = 640, ny: int = 512, fov_x_deg: float = 35.5, fov_y_deg: float = 28.7,
                  center_x: float | None = None, center_y: float | None = None,
                  to_degrees: bool = True) -> xr.DataArray:
    """Viewing zenith angle of every pixel for a pinhole camera.

    Parameters
    ----------
    nx, ny : int, default 640, 512
        Detector size in pixels.
    fov_x_deg, fov_y_deg : float, default 35.5, 28.7
        Full field of view (deg).
    center_x, center_y : float, optional
        Pixel coordinates of the optical axis. Default: the array centre.
    to_degrees : bool, default True
        Return degrees, else radians.

    Returns
    -------
    xarray.DataArray
        ``VZA`` (x, y), the angle between each pixel's view direction and
        the optical axis.
    """
    x = np.arange(nx)
    y = np.arange(ny)
    if center_x is None:
        center_x = (nx - 1) / 2
    if center_y is None:
        center_y = (ny - 1) / 2

    tan_half_x = np.tan(np.deg2rad(fov_x_deg / 2))
    tan_half_y = np.tan(np.deg2rad(fov_y_deg / 2))
    xx = (x - center_x) / (nx / 2)
    yy = (y - center_y) / (ny / 2)
    tan_theta_x = xx * tan_half_x
    tan_theta_y = yy * tan_half_y
    tan_theta = np.sqrt(tan_theta_x[:, None] ** 2 + tan_theta_y[None, :] ** 2)
    vza = np.arctan(tan_theta)
    if to_degrees:
        vza = np.rad2deg(vza)

    return xr.DataArray(
        vza, dims=('x', 'y'), name='VZA',
        attrs=dict(units='deg' if to_degrees else 'rad',
                   description='View zenith angle from optical axis (analytic pinhole model)'),
    )

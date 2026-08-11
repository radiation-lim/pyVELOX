# velox_tools/geometry.py
"""Quick analytic viewing-geometry helpers.

For calibrated per-pixel viewing angles, prefer `VELOX_viewing_angles.nc`
(used by `processing.project`) or the per-channel VDC calibration used by
`georef_paulr` -- both come from real optical/geometric calibration, not
an idealized pinhole assumption. `make_vza_map` here is the quick,
dependency-free analytic version several notebooks kept rewriting from
scratch; use it for quick looks, not for anything where sub-pixel angular
accuracy matters.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def make_vza_map(nx: int = 640, ny: int = 512, fov_x_deg: float = 35.5, fov_y_deg: float = 28.7,
                  center_x: float | None = None, center_y: float | None = None,
                  to_degrees: bool = True) -> xr.DataArray:
    """Analytic per-pixel view-zenith-angle map for a pinhole camera model.

    Parameters
    ----------
    nx, ny : int
        Detector size in pixels (640x512 for VELOX).
    fov_x_deg, fov_y_deg : float
        Full field of view (VELOX: 35.5 x 28.7 deg).
    center_x, center_y : float, optional
        Optical-axis pixel coordinates. Defaults to the array center.
    to_degrees : bool
        Return degrees (default) or radians.

    Returns
    -------
    xr.DataArray, dims ('x', 'y'), the view-zenith angle from the optical
    axis at each pixel.
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

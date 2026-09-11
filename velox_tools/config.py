# velox_tools/config.py
"""Paths to the input data.

Settings are read from a ``config.yaml`` in the working directory or the
closest parent directory that has one (see :func:`find_config`); anything
it doesn't set falls back to the defaults of :class:`DataConfig`::

    # config.yaml
    data_root: /path/to/campaign/archive
"""
import os
from typing import Optional

import yaml
from pydantic import BaseModel, Field

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# campaign archives on the server, and on Windows with /projekt_agmwend mounted as P:
_DATA_ROOTS = ['/projekt_agmwend/data', 'P:/data']


def _default_data_root():
    if 'VELOX_DATA_ROOT' in os.environ:
        return os.environ['VELOX_DATA_ROOT']
    return next((p for p in _DATA_ROOTS if os.path.isdir(p)), _DATA_ROOTS[0])


class DataConfig(BaseModel):
    """Paths to the input data.

    Attributes
    ----------
    viewing_angles : str
        Per-pixel viewing angles used by
        :func:`velox_tools.processing.project`. Default: the file shipped
        with the package.
    data_root : str
        Root of the campaign archive (the directory holding ``HALO-AC3/``
        and ``PERCUSION/``), used by :mod:`velox_tools.campaign`. Default:
        the ``VELOX_DATA_ROOT`` environment variable, else
        ``/projekt_agmwend/data`` or ``P:/data``, whichever exists.
    config_file : str or None
        The ``config.yaml`` the values were read from, if any.
    """
    viewing_angles: str = Field(
        default=os.path.join(_DATA_DIR, 'VELOX_viewing_angles.nc'),
        description="Path to the VELOX viewing angles dataset",
    )
    data_root: str = Field(
        default_factory=_default_data_root,
        description="Root of the campaign archives (HALO-AC3/, PERCUSION/), used by velox_tools.campaign. "
                    "Defaults to VELOX_DATA_ROOT, else the first of _DATA_ROOTS that exists; set it in "
                    "config.yaml for any other mount",
    )
    config_file: Optional[str] = Field(default=None, description="config.yaml these values were read from, if any")


def find_config(config_file: str = "config.yaml") -> Optional[str]:
    """Find a config file in the working directory or its parents.

    Jupyter runs notebooks in their own folder, so a ``config.yaml`` in the
    repository root is still found from ``notebooks/``.

    Parameters
    ----------
    config_file : str, default 'config.yaml'
        File name to look for.

    Returns
    -------
    str or None
        Path of the closest match, or None if there is none.
    """
    folder = os.getcwd()
    while True:
        path = os.path.join(folder, config_file)
        if os.path.exists(path):
            return path
        if os.path.dirname(folder) == folder:
            return None
        folder = os.path.dirname(folder)


def load_config(config_file: str = "config.yaml") -> DataConfig:
    """Load the configuration.

    Parameters
    ----------
    config_file : str, default 'config.yaml'
        File name to look for (see :func:`find_config`).

    Returns
    -------
    DataConfig
        Values from the file, defaults for everything it doesn't set (or
        for everything if there is no file).
    """
    path = find_config(config_file)
    if path is None:
        return DataConfig()
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}  # a config.yaml with only comments loads as None
    return DataConfig(**cfg_dict, config_file=path)

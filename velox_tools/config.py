# velox_tools/config.py

import os
import yaml
from pydantic import BaseModel, Field

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class DataConfig(BaseModel):
    nav_data: str = Field(
        default=os.path.join(_DATA_DIR, 'HALO_nav.nc'),
        description="Path to the HALO navigation dataset (including pitch, roll, alt, lat, lon)",
    )
    viewing_angles: str = Field(
        default=os.path.join(_DATA_DIR, 'VELOX_viewing_angles.nc'),
        description="Path to the VELOX viewing angles dataset",
    )
    # add more dataset paths or configuration parameters as needed


def load_config(config_file: str = "config.yaml") -> DataConfig:
    """Load configuration from a YAML file.

    `config_file` is resolved relative to the current working directory
    (so you can drop a `config.yaml` next to a script/notebook to override
    paths for that run) -- unlike the *defaults* above, which resolve
    relative to this package regardless of where you run from, since a
    silently CWD-dependent default was the whole reason a hardcoded
    absolute path crept into processing.py in the first place.
    """
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return DataConfig(**cfg_dict)
    else:
        # Return defaults if config file is not found
        return DataConfig()

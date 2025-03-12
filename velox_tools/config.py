# velox_tools/config.py

import os
import yaml
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    nav_data: str = Field(default="data/HALO_nav.nc", description="Path to the HALO navigation dataset (including pitch, roll, alt, lat, lon)")
    viewing_angles: str = Field(default="data/VELOX_viewing_angles.nc", description="Path to the VELOX viewing angles dataset")
    # add more dataset paths or configuration parameters as needed

def load_config(config_file: str = "config.yaml") -> DataConfig:
    """Load configuration from a YAML file."""
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            cfg_dict = yaml.safe_load(f)
        return DataConfig(**cfg_dict)
    else:
        # Return defaults if config file is not found
        return DataConfig()

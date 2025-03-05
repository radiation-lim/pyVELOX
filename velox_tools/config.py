# velox_tools/config.py

import os
import yaml
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    xr_HALO: str = Field(default="data/xr_HALO.nc", description="Path to the xr_HALO dataset")
    velox_data: str = Field(default="data/velox_data.nc", description="Path to the VELOX dataset")
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

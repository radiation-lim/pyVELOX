![pyVELOX logo](logo.png)

# pyVELOX

This package is still very much in development, expect changes, bugs, and missing features.
Our vision is to create a codebase for dealing with the VELOX thermal imager. This includes reading, processing, and visualizing the data.

## Installation

To install the package, clone the repository and run the following command in the root directory:

```bash
pip install .
```

To run the processing tools, you will need to add HALO navigation file, including the 'lat', 'lon', 'alt', 'pitch', 'roll', and 'heading' data variables.
Add this file as `HALO_nav.nc` in the `data` directory. 

## Usage

For usage examples, find the jupyter notebooks in the `notebooks` directory:
    - `notebooks\pushbroom.ipynb` - Example of how to create pushbroom images from the raw data
    - `notebooks\georeff.ipynb` - Example of how to georeference the raw data

## How do I get the data?

The data is not included in this repository. You can access the data [here](https://doi.pangaea.de/10.1594/PANGAEA.963401).
CAUTION: The datasets are large (~20GB) and may take a while to download.

## Processing tools for 2D thermal imagery data form the VELOX thermal infrared camera


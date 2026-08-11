from setuptools import setup, find_packages

setup(
    name="velox_tools",
    version="0.2.0",
    description="A package for processing thermal imagery data from VELOX",
    author="Joshua Müller",
    author_email="joshua.mueller@uni-leipzig.de",
    packages=find_packages(include=['velox_tools', 'velox_tools.*']),
    install_requires=[
        "numpy",
        "xarray",
        "pandas",
        "PyYAML",
        "pydantic",
        "haversine",
        "tqdm",
        "matplotlib",
        "dask",
        "distributed",
        "netcdf4",
        "scipy",
        "mounttree",  # required by velox_tools.georef_paulr
    ],
    package_data={
        "velox_tools": [
            "notebooks/*.ipynb",
            "data/*.nc",
            "data/*.yaml",
            "data/georef_paulr/*.nc",
            "data/georef_paulr/*.yaml",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

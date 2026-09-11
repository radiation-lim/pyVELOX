from setuptools import setup, find_packages

setup(
    name="velox_tools",
    version="0.3.0",
    description="Processing tools for the VELOX airborne thermal infrared imager",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/radiation-lim/pyVELOX",
    license="MIT",
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
            "notebooks/[0-9]*.ipynb",
            "data/VELOX_viewing_angles.nc",
            "data/correction_table_v1.nc",
            "data/georef_paulr/*.nc",
            "data/georef_paulr/*.yaml",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

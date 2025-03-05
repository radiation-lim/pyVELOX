from setuptools import setup, find_packages

setup(
    name="velox_tools",
    version="0.1.2",
    description="A package for processing thermal imagery data from VELOX",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=['velox_tools', 'velox_tools.*']),
    install_requires=[
        "numpy",
        "xarray",
        "pandas",
        "PyYAML",
        "pydantic",
    ],
    package_data={
        "velox_tools": ["notebooks/*.ipynb"],  # Include notebooks
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

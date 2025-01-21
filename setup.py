from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="resens",
    version="0.4.3.18",
    description="Raster Processing package for Remote Sensing and Earth Observation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nargyrop.com",
    author="Nikos Argyropoulos",
    author_email="n.argiropgeo@gmail.com",
    license="MIT",
    packages=["resens"],
    package_dir={"resens": "resens"},
    python_requires=">=3.8",
    zip_safe=False,
    install_requires=[
        "GDAL>=3",
        "geopandas>=0.11.1",
        "numpy>=1.23.4",
        "opencv-python>=4.6.0.66",
        "setuptools>=69.5.1",
        "wheel>=0.43.0",
    ],
)

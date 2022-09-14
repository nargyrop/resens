from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="resens",
    version="0.4.3.4",
    description="Raster Processing package for Remote Sensing and Earth Observation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nargyrop.com",
    author="Nikos Argyropoulos",
    author_email="n.argiropgeo@gmail.com",
    license="MIT",
    packages=["resens"],
    package_dir={"resens": "resens"},
    python_requires=">=3.9",
    zip_safe=False,
    install_requires=[
        "numpy",
        "GDAL>=3.*",
        "opencv-python"
    ]
)

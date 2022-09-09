from pathlib import Path

from setuptools import find_packages, setup

# Check for dependencies
install_requires = []
try:
    import numpy
except ImportError:
    install_requires.append("numpy")
try:
    try:
        import gdal
    except ImportError:
        from osgeo import gdal
except ImportError:
    install_requires.append("GDAL>=3.*")
try:
    import cv2
except ImportError:
    install_requires.append("opencv-python")


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="resens",
    version="0.4.2",
    description="Raster Processing package for Remote Sensing and Earth Observation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.nargyrop.com",
    author="Nikos Argyropoulos",
    author_email="n.argiropgeo@gmail.com",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    zip_safe=False,
    install_requires=install_requires
)

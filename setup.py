from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
      name='resens',
      version='0.2.3',
      description='Raster Processing package for Remote Sensing and Earth Observation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://www.nargyrop.com',
      author='Nikos Argyropoulos',
      author_email='n.argiropeo@gmail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
            'gdal', 'numpy', 'opencv-python'
      ],
      python_requires='>=3.7',
      zip_safe=False
)

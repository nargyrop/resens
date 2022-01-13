from setuptools import setup, find_packages

setup(
      name='pyeo',
      version='0.2',
      description='Raster Processing package for Remote Sensing and Earth Observation',
      url='https://www.nargyrop.com',
      author='Nikos Argyropoulos',
      author_email='n.argiropeo@gmail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
            'gdal', 'numpy', 'opencv-python'
      ],
      python_requires='>=3.8',
      zip_safe=False
)

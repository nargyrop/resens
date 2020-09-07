from setuptools import setup

setup(name='pyna',
      version='0.2',
      description='Nikos Argyropoulos EO python package',
      url='https://www.nargyrop.com',
      author='Nikos Argyropoulos',
      author_email='n.argiropeo@gmail.com',
      license='',
      packages=['pyna'],
      install_requires=['gdal', 'numpy', 'opencv-contrib-python'],
      python_requires='>=3.6.5',
      zip_safe=False)

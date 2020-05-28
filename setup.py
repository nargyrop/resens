from setuptools import setup

setup(name='pyscisys',
      version='0.2',
      description='Scisys EO python package',
      url='http://spcbs-svrep0/argyropoulos_n/pyscisys.git',
      author='Nikos Argyropoulos',
      author_email='nikos.argyropoulos@scisys.co.uk',
      license='',
      packages=['pyscisys'],
      install_requires=['gdal', 'numpy', 'opencv-contrib-python-nonfree'],
      python_requires='>=3.6.5',
      zip_safe=False)

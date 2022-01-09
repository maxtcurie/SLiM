from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('Dispersion_cython.pyx'))


#Running command
'''
python cython_setup.py build_ext --inplace
'''
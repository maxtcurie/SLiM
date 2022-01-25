from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('Cython_A_maker_V2.pyx'))

#Running command
'''
python cython_setup_numba.py build_ext --inplace
'''
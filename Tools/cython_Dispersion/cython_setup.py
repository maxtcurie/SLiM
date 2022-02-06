from distutils.core import setup
from Cython.Build import cythonize

#setup(ext_modules = cythonize('Cython_A_maker.pyx'))
setup(ext_modules = cythonize('Cython_Dispersion.pyx'))
setup(ext_modules = cythonize('Cython_Dispersion_0th_order.pyx'))
#Running command
'''
python cython_setup.py build_ext --inplace
'''
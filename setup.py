from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="power_spectrum_estimator_py._dft_density_k_pyx",
        sources=["power_spectrum_estimator_py/_dft_density_k.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="power_spectrum_estimator_py.assignment._ngp_pyx",
        sources=["power_spectrum_estimator_py/assignment/_ngp.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="power_spectrum_estimator_py.assignment._cic_pyx",
        sources=["power_spectrum_estimator_py/assignment/_cic.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="power_spectrum_estimator_py.assignment._tsc_pyx",
        sources=["power_spectrum_estimator_py/assignment/_tsc.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="power_spectrum_estimator_py",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)

import numpy as np
from ._dft_density_k_pyx import _get_dft_density_k_cython # type: ignore

def _get_dft_density_k(pos: np.ndarray, Ngrid: int, Np: int, L: float, mass: np.ndarray = None):
    """get power spectrum using dft"""

    _pos = np.array(pos, dtype=np.float64)
    density_k, kmags = _get_dft_density_k_cython(_pos, Ngrid, Np, L, mass)
    return density_k, kmags
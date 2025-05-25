from ._ngp_pyx import _NGP_cython # type: ignore
from ._cic_pyx import _CIC_cython # type: ignore
from ._tsc_pyx import _TSC_cython # type: ignore
import numpy as np

class ParticleAssigner():
    def __init__(self, L: float, Ngrid: int, Np: int):
        """
        Arguments
        ---------
        L : float
            Comoving box size of the simulation volume in [Mpc/h] units.

        Ngrid : int
            Number of grid cells per spatial dimension.

        Np: int
            Number of particles per spatial dimension.
        """
        self.L = L
        self.Ngrid = Ngrid
        self.Np = Np
        self.H = L / Ngrid
    
    def assign(self, pos:np.ndarray, mass:np.ndarray = None)-> np.ndarray:
        """
        Projects discrete particle masses onto a regular grid, generating the overdensity field δ(x) in configuration space.

        Arguments
        ---------
        pos: np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        Return
        ------
        density_r: np.ndarray[float], shape (Ngrid, Ngrid, Ngrid)
            Dimensionless overdensity field δ(x).
        """
        pass

    def _wrap_periodic(self, pos: np.ndarray) -> np.ndarray:
        """use periodic condition"""
        return pos % self.L

class NGP(ParticleAssigner):
    """Nearest Grid Point"""
    def assign(self, pos:np.ndarray, mass:np.ndarray = None)-> np.ndarray:
        pos = self._wrap_periodic(pos)
        pos = pos.astype(np.float32)
        density_r = _NGP_cython(pos, self.Ngrid, self.Np, self.H, mass)
        return density_r
    
class CIC(ParticleAssigner):
    """Cloud In Cell"""
    def assign(self, pos:np.ndarray, mass:np.ndarray = None)-> np.ndarray:
        pos = self._wrap_periodic(pos)
        pos = pos.astype(np.float32)
        density_r = _CIC_cython(pos, self.Ngrid, self.Np, self.H, mass)
        return density_r
    
class TSC(ParticleAssigner):
    """Triangular Shaped Cloud"""
    def assign(self, pos:np.ndarray, mass:np.ndarray = None)-> np.ndarray:
        pos = self._wrap_periodic(pos)
        pos = pos.astype(np.float32)
        density_r = _TSC_cython(pos, self.Ngrid, self.Np, self.H, mass)
        return density_r
    
def _mass_assign(L: float, Ngrid: int, Np: int, pos: np.ndarray, mass: np.ndarray = None, option="NGP") -> np.ndarray:
        
        match option:
            case "NGP":
                assigner = NGP(L, Ngrid, Np)
                density_r = assigner.assign(pos, mass)
            case "CIC":
                assigner = CIC(L, Ngrid, Np)
                density_r = assigner.assign(pos, mass)
            case "TSC":
                assigner = TSC(L, Ngrid, Np)
                density_r = assigner.assign(pos, mass)

        density_r /= np.mean(density_r)
        density_r -= 1.0
        return density_r
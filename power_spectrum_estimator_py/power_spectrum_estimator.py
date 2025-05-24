import numpy as np
from .assignment._mass_assign import _mass_assign
from ._dft_density_k import _get_dft_density_k
from ._pk_estimator import _pk_estimator

class Power_spectrum_estimator:
    def __init__(self, Np: int, Ngrid: int, L: float):
        """
        Power spectrum estimator

        Arguments
        ---------
        Np : int
            Number of particles per spatial dimension.

        Ngrid : int
            Number of grid cells per spatial dimension.
        
        L : float
            Comoving box size of the simulation volume in [Mpc/h] units.
        """
        self.Ngrid = Ngrid
        self.Np = Np
        self.L = L
        self.H = L / Ngrid
        self.ks = np.fft.fftfreq(self.Ngrid, self.H) * 2 * np.pi
        self.ks_z = np.fft.rfftfreq(self.Ngrid, self.H) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(self.ks, self.ks, self.ks_z)
        self.kmags = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)

    def mass_assign(self, pos: np.ndarray, mass: np.ndarray = None, option="NGP") -> np.ndarray:
        """
        Mass assignment: paint particles to grids

        Arguments
        ---------
        pos : np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        option : 
            mass assignment scheme: NGP/CIC/TSC.
        
        Return
        ------
        density_r : np.ndarray[float], shape (Ngrid, Ngrid, Ngrid)
            Dimensionless overdensity field δ(x).
        """
        
        return _mass_assign(self.L, self.Ngrid, self.Np, pos, mass, option)

    def get_density_k_dft(self, pos: np.ndarray, mass: np.ndarray = None) -> np.ndarray:
        """
        get density_k with dft

        Arguments
        ---------
        pos : np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        Returns
        -------
        density_k : np.ndarray[complex], shape (Ngrid, Ngrid, Ngrid//2 + 1)
            Complex density contrast field in Fourier space.

        kmags : np.ndarray[float], shape (Ngrid, Ngrid, Ngrid//2 + 1)
            Magnitudes of comoving wavevectors in [h/Mpc] units.
        """
        density_k, kmags = _get_dft_density_k(pos, self.Ngrid, self.Np, self.L, mass)
        return density_k, kmags

    def get_density_k_fft(self, pos: np.ndarray, mass: np.ndarray = None, assigner="NGP", anti_aliasing=None):
        """
        get density_k with fft

        Arguments
        ----------
        pos : np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        assigner : 
            mass assign scheme : NGP/CIC/TSC.

        anti_aliasing : 
            "interlacing": interlaced pk  / None: not interlaced pk.


        Returns
        --------
        density_k : np.ndarray[complex], shape (Ngrid, Ngrid, Ngrid//2 + 1)
            Complex density contrast field in Fourier space.
        
        """
        density_k = np.zeros((self.Ngrid, self.Ngrid, self.Ngrid//2 + 1),dtype=complex)
        density_r1 = _mass_assign(self.L, self.Ngrid, self.Np, pos, mass, assigner)
        density_k1 = np.fft.rfftn(density_r1)
        density_k1 *= self.H**3

        if anti_aliasing == None:
            return density_k1
        if anti_aliasing == "interlacing":
            pos = (pos - self.H/2) % self.L
            density_r2 = _mass_assign(self.L, self.Ngrid, self.Np, pos, mass, assigner)
            density_k2 = np.fft.rfftn(density_r2)
            density_k2 *= self.H**3
            density_k2 *= np.exp(-1j*(self.kx + self.ky + self.kz)*self.H/2)
            density_k = 0.5 * (density_k1 + density_k2)
        
            return density_k
        
    def get_pk_dft(self, pos: np.ndarray, mass: np.ndarray = None) -> np.ndarray:
        """
        power spectrum estimator with given positions and masses using dft

        Arguments
        ---------
        pos : np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        Returns
        -------
        k_bins : np.ndarray[float], shape (N_bins,)
            Comoving wavenumbers in [h/Mpc] units, representing the center of each k_bin in Fourier space. Bins are constructed 
            according to the minimum fundamental mode (k_min = 2π/L) and maximum Nyquist frequency (k_max = π*Ngrid/L).

        Pk : np.ndarray[float], shape (N_bins,)
            Matter power spectrum P(k) in [(Mpc/h)^3] units.

        modes : np.ndarray[int], shape (N_bins,)
            Number of modes in each k_bin
        
        """
        density_k, kmags = self.get_density_k_dft(pos, mass)
        return self.get_pk(density_k, kmags)
    
    def get_pk_fft(self, pos: np.ndarray, mass: np.ndarray = None, assigner="NGP", anti_aliasing=None, compensation=None):
        """
        power spectrum estimator with given positions and masses using fft

        Arguments
        ----------
        pos : np.ndarray[float], shape (N_particle, 3)
            Comoving particle positions in [L] units, where L is the box size.
        
        mass : np.ndarray[float], optional, shape (N_particle)
            Particle masses. If None (default), each particle weights 1.0.

        assigner : 
            mass assign scheme : NGP/CIC/TSC.

        anti_aliasing : 
            "interlacing": interlaced pk  / None: not interlaced pk.

        compensation : 
            "deconvolving": deconvolved pk / None: not deconvolved pk.
        
        Returns
        -------
        k_bins : np.ndarray[float], shape (N_bins,)
            Comoving wavenumbers in [h/Mpc] units, representing the center of each k_bin in Fourier space. Bins are constructed 
            according to the minimum fundamental mode (k_min = 2π/L) and maximum Nyquist frequency (k_max = π*Ngrid/L).

        Pk : np.ndarray[float], shape (N_bins,)
            Matter power spectrum P(k) in [(Mpc/h)^3] units.

        modes : np.ndarray[int], shape (N_bins,)
            Number of modes in each k_bin
        
        """
        density_k = self.get_density_k_fft(pos, mass, assigner, anti_aliasing)
        if compensation == None:
            return self.get_pk(density_k, self.kmags)
        if compensation == "deconvolving":
            density_k = self.deconvolve(density_k, assigner)
            return self.get_pk(density_k, self.kmags)

    def get_pk(self, density_k: np.ndarray, kmags: np.ndarray):
        """
        power spectrum estimator with given density_k

        Arguments
        ---------
        density_k: np.ndarray[complex], shape (Ngrid, Ngrid, Ngrid//2 + 1)
            Complex density contrast field in Fourier space.

        kmags: np.ndarray[float], shape (Ngrid, Ngrid, Ngrid//2 + 1)
            Magnitudes of comoving wavevectors in [h/Mpc] units.

        Returns
        -------
        k_bins : np.ndarray[float], shape (N_bins,)
            Comoving wavenumbers in [h/Mpc] units, representing the center of each k_bin in Fourier space. Bins are constructed 
            according to the minimum fundamental mode (k_min = 2π/L) and maximum Nyquist frequency (k_max = π*Ngrid/L).

        Pk : np.ndarray[float], shape (N_bins,)
            Matter power spectrum P(k) in [(Mpc/h)^3] units.

        modes : np.ndarray[int], shape (N_bins,)
            Number of modes in each k_bin
        
        """      
        return _pk_estimator(density_k, kmags, self.L, self.Ngrid)
    
    def deconvolve(self, density_k: np.ndarray, assigner = "NGP") -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = lambda x: np.sin(x * self.H / 2 + 1e-20)/ (x * self.H / 2 + 1e-20)
            Wx = sinc(self.kx)
            Wy = sinc(self.ky)
            Wz = sinc(self.kz)

            W = Wx * Wy * Wz

            match assigner:
                case "NGP":
                    return density_k / W
                case "CIC":
                    return density_k / W**2
                case "TSC":
                    return density_k / W**3
                    



    

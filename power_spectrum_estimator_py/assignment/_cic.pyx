import numpy as np
cimport numpy as cnp

def _CIC_cython(cnp.float32_t[:, :] pos, int Ngrid, int Np, float H, mass = None):
    cdef int i, ix, iy, iz, x, y, z, nx, ny, nz
    cdef float px, py, pz, weight
    cdef cnp.float32_t[:, :, :] density_r = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    
    if mass is None:
        mass = np.ones(Np, dtype=np.float32)
    else:
        mass = mass.astype(np.float32)

    for i in range(Np):
        px = pos[i,0] / H
        py = pos[i,1] / H
        pz = pos[i,2] / H
        x = int(np.floor(px))
        y = int(np.floor(py))
        z = int(np.floor(pz))
        
        for ix in [0, 1]:
            for iy in [0, 1]:
                for iz in [0, 1]:
                    nx = (x + ix) % Ngrid
                    ny = (y + iy) % Ngrid
                    nz = (z + iz) % Ngrid
                    weight = (
                        (1.0 - abs(x + ix - px)) * 
                        (1.0 - abs(y + iy - py)) * 
                        (1.0 - abs(z + iz - pz))) * mass[i]
                    density_r[nx, ny, nz] += weight 
    
    return np.asarray(density_r)
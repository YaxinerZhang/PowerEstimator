import numpy as np
cimport numpy as cnp

def _NGP_cython(cnp.float32_t[:, :] pos, int Ngrid, int Np, float H, mass = None):
    cdef int i, x, y, z, nx, ny, nz
    cdef float px, py, pz
    cdef cnp.float32_t[:, :, :] density_r = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)

    if mass is None:
        mass = np.ones(Np, dtype=np.float32)
    else:
        mass = mass.astype(np.float32)

    for i in range(len(pos)):
        px = pos[i][0] / H
        py = pos[i][1] / H
        pz = pos[i][2] / H
        x = int(np.round(px))
        y = int(np.round(py))
        z = int(np.round(pz))
       
        nx = x % Ngrid
        ny = y % Ngrid
        nz = z % Ngrid
        density_r[nx, ny, nz] += mass[i] 
    return np.asarray(density_r)
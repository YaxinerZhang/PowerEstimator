import numpy as np
cimport numpy as cnp

def _TSC_cython(cnp.float64_t[:, :] pos, int Ngrid, int Np, float H, mass = None):
    cdef int i, ix, iy, iz, x, y, z, nx, ny, nz
    cdef float px, py, pz, dx, dy, dz, pmass, weight
    cdef list wx = [], wy = [], wz = []
    cdef cnp.float64_t[:, :, :] density_r = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    
    for i in range(len(pos)):
        px = pos[i][0] / H
        py = pos[i][1] / H
        pz = pos[i][2] / H
        x = int(np.rint(px))
        y = int(np.rint(py))
        z = int(np.rint(pz))

        dx = x - px
        dy = y - py
        dz = z - pz 

        wx = _WTSC(dx)
        wy = _WTSC(dy)
        wz = _WTSC(dz)


        if mass == None:
            pmass = 1.0 
        else:
            pmass = mass[i]
        
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    nx = (x + ix) % Ngrid
                    ny = (y + iy) % Ngrid
                    nz = (z + iz) % Ngrid
                    weight = wx[ix+1] * wy[iy+1] * wz[iz+1] * pmass
                    density_r[nx, ny, nz] += weight 
      
    return np.asarray(density_r)


def _WTSC(diff):
    w_center = 0.75 - diff**2
    w_before = 0.5 * (0.5+diff)*(0.5+diff)
    w_after = 0.5 * (0.5-diff)*(0.5-diff)
    
    return [w_before, w_center, w_after]

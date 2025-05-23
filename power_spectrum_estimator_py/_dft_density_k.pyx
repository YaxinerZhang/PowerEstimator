import numpy as np
cimport numpy as cnp

cnp.import_array()

def _get_dft_density_k_cython(cnp.float64_t[:, ::1] pos, int N, int Np, float L,  mass = None):
    cdef:
        int i, j, k
        float kx, ky, kz, kmag
        complex sum_exp
        float k_f = 2 * np.pi / L
        float dr = L / Np
        list density_k = []
        list kmags = []
        cnp.ndarray[cnp.complex128_t, ndim=1] phase

    cdef:
        cnp.ndarray[cnp.float64_t, ndim=1] x = np.asarray(pos[:,0])
        cnp.ndarray[cnp.float64_t, ndim=1] y = np.asarray(pos[:,1])
        cnp.ndarray[cnp.float64_t, ndim=1] z = np.asarray(pos[:,2])

    if mass == None:
        mass = np.ones(Np**3, dtype=np.float64)
    
    for i in range(N):
        kx = (-N//2 + i) * k_f
        for j in range(N):
            ky = (-N//2 + j) * k_f
            for k in range(N//2 + 1):  
                kz = k * k_f
                kmag = np.sqrt(kx**2 + ky**2 + kz**2)
                phase = -1j * (kx * x + ky * y + kz * z) 
                sum_exp = np.sum(np.exp(phase) * mass)
                density_k.append(dr**3 * sum_exp)
                kmags.append(kmag)
    
    return np.asarray(density_k), np.asarray(kmags)
    


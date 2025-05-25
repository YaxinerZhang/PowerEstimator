import numpy as np

def _pk_estimator(density_k: np.ndarray, kmags: np.ndarray, L: float, Ngrid:int):

    k_f = 2 * np.pi / L
    k_vals = np.arange(k_f, k_f * Ngrid/2, k_f)
    V = L**3
    Pk = []
    modes = []
    for i in range(len(k_vals)):
        pk_bin = 0
        k_bin_indices = np.where((kmags > k_vals[i] - k_f/2) & (kmags <= k_vals[i] + k_f/2))
        pk_bin = 1/ V * np.mean(np.abs(density_k[k_bin_indices])**2)
        Pk.append(pk_bin)
        modes.append(len(k_bin_indices[0]))
    return k_vals, np.asarray(Pk), np.asarray(modes)
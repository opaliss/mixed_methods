"""module with Hermite functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: Oct 23rd, 2025
"""
import numpy as np
import scipy



def sw_psi_hermite(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach)

    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, AW aw_hermite polynomial of degree n evaluated at xi
    """
    # scaled velocity coordinate
    xi = (v - u_s) / alpha_s
    # iteratively compute psi_{n}(xi)
    if n == 0:
        return np.exp(-0.5 * (xi ** 2)) / np.sqrt(np.sqrt(np.pi))
    if n == 1:
        return np.exp(-0.5 * (xi ** 2)) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
    else:
        psi = np.zeros((n + 1, len(xi)))
        psi[0, :] = np.exp(-0.5 * (xi ** 2)) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-0.5 * (xi ** 2)) * (2 * xi) / np.sqrt(2 * np.sqrt(np.pi))
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj + 1) / 2)
            psi[jj + 1, :] = (alpha_s * np.sqrt(jj / 2) * psi[jj - 1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]



"""module with Legendre functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np
from operators.universal_functions import nu_func



def xi_legendre(n, v, v_a, v_b):
    """AW Hermite basis function (iterative approach)

    :param v_a: float, velocity lower limit
    :param v_b, float, velocity upper limit
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, Legendre basis function of degree n on a grid v
    """
    # scaled velocity coordinate
    eta = (2 * v - (v_a + v_b))/(v_b - v_a)
    # iteratively compute psi_{n}(xi)
    if n == 0:
        return np.sqrt(2*n + 1) * 1
    if n == 1:
        return np.sqrt(2*n + 1) * eta
    else:
        xi = np.zeros((n+1, len(v)))
        xi[0, :] = 1
        xi[1, :] = np.sqrt(3) * eta
        for jj in range(1, n):
            xi[jj+1, :] = ((2*n + 1)*eta * xi[jj, :] - jj*xi[jj-1, :]) / (n+1) * np.sqrt(2*n+1)
    return xi[n, :]



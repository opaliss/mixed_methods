"""module with universal functions used for Hermite and Legendre formulations

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np


def get_D_inv(Nx, D):
    """inverse of derivative D matrix

    :param Nx: int, number of spatial grid points
    :param D: 2d array (matrix), finite difference derivative matrix
    :return: 2d array (matrix), inverse of D
    """
    mat = np.zeros((Nx + 1, Nx + 1))
    mat[:-1, :-1] = D.toarray()
    mat[-1, :-1] = np.ones(Nx)
    mat[:-1, -1] = np.ones(Nx)
    return np.linalg.inv(mat)


def nu_func(n, Nv):
    """coefficient for hyper-collisions

    :param n: int, index of spectral term
    :param Nv: int, total number of Hermite spectral expansion coefficients
    :return: float, coefficient for hyper-collisions
    """
    return n * (n - 1) * (n - 2) / (Nv - 1) / (Nv - 2) / (Nv - 3)

"""update the f0 and df distributions.

Author: Opal Issan (oissan@ucsd.edu)
Date: Oct 24th, 2025
"""

import numpy as np


def reprojection_aw_hermite_and_legendre(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, J):
    """

    :param cutoff: int, has to be less than Nv_e1.
    :param Nx: int, spatial resolution
    :param Nv_e1: int, velocity resolution "bulk"
    :param Nv_e2: int, velocity resolution "bump"
    :param y_curr: 1d array, solution from previous timestep
    :param v_a: float, lower bound of the velocity coordinate for Legendre
    :param v_b: float, upper bound of the velocity coordinate for Legendre
    :param J: 2d array, matrix with projection of aw hermite and legendre basis J_[n,m] = int psi_n xi_m
    :return:
    """
    new_solution = np.zeros(len(y_curr))
    new_solution[:Nx * cutoff] = y_curr[:Nx * cutoff]

    for m in range(Nv_e2):
        hermite_correction = np.zeros(Nx)
        for p in range(cutoff, Nv_e1):
            hermite_correction += y_curr[p * Nx:(p + 1) * Nx] * J[p, m] / (v_b - v_a)

        new_solution[Nx * Nv_e1 + m * Nx: Nx * Nv_e1 + (m + 1) * Nx] = y_curr[Nx * Nv_e1 + m * Nx: Nx * Nv_e1 + (
                m + 1) * Nx] + hermite_correction
    return new_solution


def reprojection_adaptive_in_space_aw_hermite_and_legendre(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, J):
    """

    :param cutoff:
    :param Nx:
    :param Nv_e1:
    :param Nv_e2:
    :param y_curr:
    :param v_a:
    :param v_b:
    :param J:
    :return:
    """
    new_solution = np.zeros(len(y_curr))
    new_solution[:Nx * cutoff] = y_curr[:Nx * cutoff]

    for ii in range(Nx):
        for m in range(Nv_e2):
            hermite_correction = 0
            for p in range(cutoff, Nv_e1):
                hermite_correction += y_curr[p * Nx + ii] * J[p, m, ii] / (v_b - v_a)
            new_solution[Nx * Nv_e1 + m * Nx + ii] = y_curr[Nx * Nv_e1 + m * Nx + ii] + hermite_correction
    return new_solution

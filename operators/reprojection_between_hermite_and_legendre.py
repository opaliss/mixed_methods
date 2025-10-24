"""update the f0 and df distributions.

Author: Opal Issan (oissan@ucsd.edu)
Date: Oct 24th, 2025
"""

import numpy as np


def reprojection_between_aw_hermite_and_legendre(cutoff, Nx, Nv_e1, Nv_e2, y_curr, v_a, v_b, J):
    """
    f0 = sum_{0}^{cutoff} C_{n} \psi_{n} and df = sum_{0}^{N_{L}-1} B_{m} \xi_{m}

    :param cutoff:
    :param y_curr:
    :param v_a:
    :param v_b:
    :param Nx:
    :param Nv_e1:
    :param Nv_e2:
    :param J:
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

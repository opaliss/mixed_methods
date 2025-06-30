"""module with mixed method #2 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np


def extra_term_2(LH_int_2, Nv_H, D, Nx, state_legendre, Nv_L):
    """

    :param LH_int_2:
    :param state_legendre:
    :param Nx:
    :param Nv_H:
    :param Nv_L:
    :param D:
    :return:
    """
    res = np.zeros(Nx * Nv_H)
    res[(Nv_H - 1) * Nx:] = - np.sqrt(Nv_H / 2) * D @ \
                                      ((LH_int_2[:, None] * state_legendre.reshape(Nv_L, Nx)).sum(axis=0))

    return res


def extra_term_3(LH_int_2, LH_int_3, Nv_H, Nv_L, Nx, v_b, v_a, D, state_legendre):
    """

    :param LH_int_2:
    :param LH_int_3:
    :param Nv_H:
    :param Nv_L:
    :param Nx:
    :param v_b:
    :param v_a:
    :param D:
    :param state_legendre:
    :return:
    """
    return 1 / (v_b - v_a) * np.sqrt(Nv_H / 2) \
           * np.kron(LH_int_3, D @ ((LH_int_2[:, None] * state_legendre.reshape(Nv_L, Nx)).sum(axis=0)))

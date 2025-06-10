"""module with mixed method #1 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np


def extra_term_1(LH_int, v_b, v_a, C_hermite_last, alpha, Nv_H, D, E):
    """

    :param LH_int:
    :param v_b:
    :param v_a:
    :param C_hermite_last:
    :param alpha:
    :param Nv_H:
    :param D:
    :param E:
    :return:
    """
    A = alpha * np.sqrt(Nv_H / 2) * (D @ C_hermite_last - 2 / (alpha ** 2) * E * C_hermite_last)
    return - 1 / (v_b - v_a) * np.kron(LH_int, A)




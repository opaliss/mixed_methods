"""module with mixed method #1 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np
from operators.legendre.legendre_operators import construct_f


def extra_term_1(LH_int_1, v_b, v_a, C_hermite_last, alpha, Nv_H, D, E,  Nv_L, Nx):
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
    A = alpha * np.sqrt(Nv_H / 2) * (D @ C_hermite_last + 2 / (alpha ** 2) * E * C_hermite_last)
    extra_term_2 = np.zeros(Nx * Nv_L)
    for ii in range(Nv_L):
        extra_term_2[ii*Nx: (ii+1)*Nx] = -1 / (v_b - v_a) * LH_int_1[ii] * A
    return extra_term_2


def closure_term(Nv_L, D, LH_int_complement, state, Nx, Nv_H):
    closure = np.zeros(Nv_H * Nx)
    closure[(Nv_H - 1)*Nx:] = - np.sqrt(Nv_H/2) * D  @ construct_f(state=state, Nv=Nv_L, Nx=Nx, xi=LH_int_complement)
    return closure

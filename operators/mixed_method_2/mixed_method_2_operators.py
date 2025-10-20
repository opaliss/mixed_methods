"""module with mixed method #2 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np


def extra_term_2(I_int_complement, Nv_H, D, Nx, state_legendre, Nv_L):
    """

    :param I_int_complement:
    :param state_legendre:
    :param Nx:
    :param Nv_H:
    :param Nv_L:
    :param D:
    :return:
    """
    sol_ = np.zeros(Nx * Nv_H)
    sol_[(Nv_H - 1) * Nx:] = - np.sqrt(Nv_H / 2) * summation_term(I_int_complement=I_int_complement,
                                                                  D=D, state_legendre=state_legendre, Nx=Nx, Nv_L=Nv_L)
    return sol_


def summation_term(I_int_complement, D, state_legendre, Nx, Nv_L):
    """

    :param I_int_complement:
    :param D:
    :param state_legendre:
    :param Nx:
    :param Nv_L:
    :return:
    """
    sol_ = np.zeros(Nx)
    for ii in range(Nv_L):
        sol_ += I_int_complement[ii] * state_legendre[ii * Nx: (ii + 1) * Nx]
    return  D @ sol_


def extra_term_3(I_int_complement, J_int, Nv_H, Nv_L, Nx, v_b, v_a, D, state_legendre):
    """

    :param I_int_complement:
    :param J_int:
    :param Nv_H:
    :param Nv_L:
    :param Nx:
    :param v_b:
    :param v_a:
    :param D:
    :param state_legendre:
    :return:
    """
    sol_ = np.zeros(Nx * Nv_L)
    sum_term = summation_term(I_int_complement=I_int_complement, D=D, state_legendre=state_legendre, Nx=Nx, Nv_L=Nv_L)
    for ii in range(Nv_L):
        sol_[ii * Nx: (ii + 1) * Nx] += 1 / (v_b - v_a) * np.sqrt(Nv_H / 2) * J_int[ii] * sum_term
    return sol_

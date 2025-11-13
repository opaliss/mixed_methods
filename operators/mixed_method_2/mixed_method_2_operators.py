"""module with mixed method #2 with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np
from operators.legendre.legendre_operators import boundary_term


def extra_term_1_hermite(I_int_complement, Nv_H, D, Nx, state_legendre, Nv_L):
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


def extra_term_2_hermite(E, psi, q, m, Nv, Nx, gamma, v_a, v_b, psi_dual_v_a, psi_dual_v_b, alpha):
    """compute acceleration term (nonlinear)

    :param psi_dual_v_b:
    :param psi_dual_v_a:
    :param alpha:
    :param v_b:
    :param v_a:
    :param E: 1d array, electric field on finite difference mesh
    :param psi: 1d array, vector of all coefficients
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :param gamma: float, penalty term
    :return: N(E, psi)
    """
    res_boundary = np.zeros(len(psi))
    for nn in range(0, Nv):
        if gamma != 0:
            res_boundary[nn * Nx: (nn + 1) * Nx] += -boundary_term(n=nn, gamma=gamma,
                                                                   v_b=v_b, v_a=v_a, Nx=Nx, Nv=Nv,
                                                                   psi=psi, xi_v_a=psi_dual_v_a, xi_v_b=psi_dual_v_b)
    return (res_boundary.reshape(Nv, Nx) * q / m * E).flatten() / alpha


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
    return D @ sol_


def extra_term_2_legendre(I_int_complement, J_int, Nv_H, Nv_L, Nx, v_b, v_a, D, state_legendre):
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
    sum_term = 1 / (v_b - v_a) * np.sqrt(Nv_H / 2) * summation_term(I_int_complement=I_int_complement,
                                                                    D=D, state_legendre=state_legendre,
                                                                    Nx=Nx, Nv_L=Nv_L)
    for ii in range(Nv_L):
        sol_[ii * Nx: (ii + 1) * Nx] += J_int[ii] * sum_term
    return sol_


def extra_term_3_legendre(J_int, Nv_H, Nv_L, Nx, v_b, v_a, state_legendre, psi_dual_v_b, psi_dual_v_a, alpha, gamma, E,
                          q, m):
    """

    :param J_int:
    :param Nv_H:
    :param Nv_L:
    :param Nx:
    :param v_b:
    :param v_a:
    :param state_legendre:
    :param psi_dual_v_b:
    :param psi_dual_v_a:
    :param alpha:
    :param gamma:
    :param E:
    :param q:
    :param m:
    :return:
    """
    res_boundary = np.zeros(len(state_legendre))
    for mm in range(0, Nv_L):
        for nn in range(0, Nv_H):
            res_boundary[mm * Nx: (mm + 1) * Nx] += boundary_term(n=nn, gamma=gamma,
                                                                   v_b=v_b, v_a=v_a,
                                                                   Nx=Nx, Nv=Nv_L,
                                                                   psi=state_legendre,
                                                                   xi_v_a=psi_dual_v_a, xi_v_b=psi_dual_v_b) * J_int[nn, mm]

    return (res_boundary.reshape(Nv_L, Nx) * q / m * E).flatten() / alpha
"""module with mixed (static) method with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""
import numpy as np

def charge_density_two_stream_mixed_method_0(q_e1, q_e2, q_i, alpha_e1, alpha_e2, alpha_i, C0_e1, C0_e2, C0_i):
    """charge density (right hand side of Poisson equation)

    :param q_e1: float, charge of electrons species 1
    :param q_e2: float, charge of electrons species 2
    :param q_i: float, charge of ions
    :param alpha_e1: float, hermite scaling parameter or thermal velocity of electrons species 1
    :param alpha_e2: float, hermite scaling parameter or thermal velocity of electrons species 2
    :param alpha_i: float, hermite scaling parameter or thermal velocity of ions
    :param C0_e1: 1d array, density of electrons species 1
    :param C0_e2: 1d array, density of electrons species 2
    :param C0_i: 1d array, density of ions
    :return: change density rho(x, t=t*)
    """
    return q_e1 * alpha_e1 * C0_e1 + q_e2 * alpha_e2 * C0_e2 + q_i * alpha_i * C0_i

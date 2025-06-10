"""module with mixed (static) method with bulk Hermite and beam Legendre

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""


def charge_density_two_stream_mixed_method_0(q_e, alpha_e, v_a, v_b, C0_e_hermite, C0_e_legendre):
    """charge density (right hand side of Poisson equation)

    :param q_e: float, charge of electrons
    :param alpha_e: float, Hermite scaling parameter or thermal velocity of bulk electrons
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :param C0_e_legendre: 1d array, density of electrons (beam) described with Legendre basis
    :param C0_e_hermite: 1d array, density of electrons (bulk) described with Hermite basis
    :return: change density rho(x, t=t*)
    """
    return q_e * (alpha_e * C0_e_hermite + (v_b - v_a) * C0_e_legendre) + 1


def total_mass_mixed_method_0(state, alpha_s, dx):
    """total mass of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :return: total mass of single electron and ion setup
    """
    return mass_hermite(state=state) * dx * alpha_s


def total_momentum_mixed_method_0(state, alpha_s, dx, m_s, u_s):
    """total momentum of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return momentum_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


def total_energy_k_mixed_method_0(state, alpha_s, dx, m_s, u_s):
    """total kinetic energy of single electron and ion setup

    :param state: 1d array, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


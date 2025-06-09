"""module with Hermite functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np
import scipy


def psi_hermite(n, alpha_s, u_s, v):
    """AW Hermite basis function (iterative approach)

    :param alpha_s: float, velocity scaling parameter
    :param u_s, float, velocity shifting parameter
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, AW hermite polynomial of degree n evaluated at xi
    """
    # scaled velocity coordinate
    xi = (v - u_s)/alpha_s
    # iteratively compute psi_{n}(xi)
    if n == 0:
        return np.exp(-xi ** 2) / np.sqrt(np.pi)
    if n == 1:
        return np.exp(-xi ** 2) * (2*xi)/np.sqrt(2*np.pi)
    else:
        psi = np.zeros((n+1, len(xi)))
        psi[0, :] = np.exp(-xi ** 2) / np.sqrt(np.pi)
        psi[1, :] = np.exp(-xi ** 2) * (2*xi) / np.sqrt(2*np.pi)
        for jj in range(1, n):
            factor = - alpha_s * np.sqrt((jj+1)/2)
            psi[jj+1, :] = (alpha_s * np.sqrt(jj/2) * psi[jj-1, :] + u_s * psi[jj, :] - v * psi[jj, :]) / factor
    return psi[n, :]


def A1_hermite(D, Nv):
    """A1 matrix advection term with alpha

    :param D: 2d array (matrix), finite difference derivative matrix
    :param Nv: int, Hermite spectral resolution
    :return: 2d array (matrix), A1 matrix in advection term
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        if n != 0:
            # lower diagonal
            A[n, n - 1] = np.sqrt(n / 2)
        if n != Nv - 1:
            # upper diagonal
            A[n, n + 1] = np.sqrt((n + 1) / 2)
    return -scipy.sparse.kron(A, D, format="csr")


def nonlinear_hermite(E, psi, q, m, alpha, Nv, Nx):
    """compute acceleration term (nonlinear)

    :param E: 1d array, electric field on finite difference mesh
    :param psi: 1d array, vector of all coefficients
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param alpha: float, temperature of particles
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :return: N(E, psi)
    """
    res = np.zeros(len(psi))
    for n in range(Nv):
        if n != 0:
            res[n*Nx: (n+1)*Nx] = q/m/alpha * np.sqrt(2*n) * E * psi[(n-1)*Nx: n*Nx]
    return res


def charge_density_hermite(q_e, q_i, alpha_e, alpha_i, C0_e, C0_i):
    """charge density (right hand side of Poisson equation)

    :param q_e: float, charge of electrons
    :param q_i: float, charge of ions
    :param alpha_e: float, hermite scaling parameter or thermal velocity of electrons
    :param alpha_i: float, hermite scaling parameter or thermal velocity of ions
    :param C0_e: 1d array, density of electrons
    :param C0_i: 1d array, density of ions
    :return: change density rho(x, t=t*)
    """
    return q_e * alpha_e * C0_e + q_i * alpha_i * C0_i


def mass_hermite(state):
    """mass of a single specie

    :param state: 1d array, electron or ion state
    :return: mass for the state
    """
    return np.sum(state[0, :])


def momentum_hermite(state, u_s, alpha_s):
    """momentum of a single specie

    :param state: 1d array, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: momentum for the state
    """
    return alpha_s * np.sum(state[1, :]) / np.sqrt(2) + u_s * np.sum(state[0, :])


def energy_k_hermite(state, u_s, alpha_s):
    """kinetic energy of a single specie

    :param state: 1d array, electron or ion state
    :param u_s: float, the velocity shifting parameter of species s
    :param alpha_s: float, the velocity scaling parameter of species s
    :return: kinetic energy for the state
    """
    return (alpha_s**2) / np.sqrt(2) * np.sum(state[2, :]) + np.sqrt(2) * u_s * alpha_s * np.sum(state[1, :]) \
           + (alpha_s**2/2 + u_s**2) * np.sum(state[0, :])


def total_mass_hermite(state, alpha_s, dx):
    """total mass of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :return: total mass of single electron and ion setup
    """
    return mass_hermite(state=state) * dx * alpha_s


def total_momentum(state, alpha_s, dx, m_s, u_s):
    """total momentum of single electron and ion setup

    :param state: 1d array, species s state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total momentum of single electron and ion setup
    """
    return momentum_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


def total_energy_k_hermite(state, alpha_s, dx,  m_s, u_s):
    """total kinetic energy of single electron and ion setup

    :param state: 1d array, species s  state
    :param alpha_s: float, velocity scaling of species s
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param u_s: float, velocity shifting parameter of species s
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k_hermite(state=state, alpha_s=alpha_s, u_s=u_s) * dx * alpha_s * m_s


def charge_density_two_stream_hermite(q_e1, q_e2, q_i, alpha_e1, alpha_e2, alpha_i, C0_e1, C0_e2, C0_i):
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

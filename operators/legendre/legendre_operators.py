"""module with Legendre functions and operators

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 8th, 2025
"""
import numpy as np
import scipy


def xi_legendre(n, v, v_a, v_b):
    """AW Hermite basis function (iterative approach)

    :param v_a: float, velocity lower limit
    :param v_b, float, velocity upper limit
    :param v: float or array, the velocity coordinate on a grid
    :param n: int, order of polynomial
    :return: float or 1d array, Legendre basis function of degree n on a grid v
    """
    # scaled velocity coordinate
    eta = (2 * v - (v_a + v_b)) / (v_b - v_a)
    if isinstance(eta, float):
        if n == 0:
            return np.sqrt(2 * n + 1)
        if n == 1:
            return np.sqrt(2 * n + 1) * eta
        else:
            xi = np.zeros(n + 1)
            xi[0] = 1
            xi[1] = eta
            for jj in range(1, n):
                xi[jj + 1] = ((2 * jj + 1) * eta * xi[jj] - jj * xi[jj - 1]) / (jj + 1)
            return xi[n] * np.sqrt(2 * n + 1)
    else:
        if n == 0:
            return np.sqrt(2 * n + 1) * np.ones(len(eta))
        if n == 1:
            return np.sqrt(2 * n + 1) * eta
        else:
            xi = np.zeros((n + 1, len(v)))
            xi[0, :] = np.ones(len(v))
            xi[1, :] = eta
            for jj in range(1, n):
                xi[jj + 1, :] = ((2 * jj + 1) * eta * xi[jj, :] - jj * xi[jj - 1, :]) / (jj + 1)
            return xi[n, :] * np.sqrt(2 * n + 1)


def A1(D, Nv, v_a, v_b):
    """A1 matrix advection term with sigma

    :param D: 2d array (matrix), finite difference derivative matrix
    :param Nv: int, Hermite spectral resolution
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :return: 2d array (matrix), A1 matrix in advection term
    """
    A = np.zeros((Nv, Nv))
    for n in range(Nv):
        if n != 0:
            # lower diagonal
            A[n, n - 1] = sigma_v1(n=n, v_a=v_a, v_b=v_b)
        if n != Nv - 1:
            # upper diagonal
            A[n, n + 1] = sigma_v1(n=n + 1, v_a=v_a, v_b=v_b)
    return -scipy.sparse.kron(A, D, format="csr")


def sigma_v1(n, v_a, v_b):
    """sigma(n)

    :param n: int, index of sigma
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :return: sigma(n)
    """
    if n >= 1:
        return (v_b - v_a) * 0.5 * n / np.sqrt((2 * n + 1) * (2 * n - 1))
    else:
        return 0


def nonlinear_full(E, psi, q, m, v_a, v_b, Nv, Nx, gamma):
    """compute acceleration term (nonlinear)

    :param E: 1d array, electric field on finite difference mesh
    :param psi: 1d array, vector of all coefficients
    :param q: float, charge of particles
    :param m: float, mass of particles
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :param Nx: int, grid size in space
    :param Nv: int, spectral resolution in velocity
    :param gamma: float, penalty term
    :return: N(E, psi)
    """
    res = np.zeros(len(psi))
    for nn in range(Nv):
        if nn != 0:
            for ii in range(nn):
                sig = sigma_v2(n=nn, i=ii, v_a=v_a, v_b=v_b)
                res[nn * Nx: (nn + 1) * Nx] += psi[ii * Nx: (ii + 1) * Nx] * sigma_v2(n=nn, i=ii, v_a=v_a, v_b=v_b)
        if gamma != 0:
            res[nn * Nx: (nn + 1) * Nx] += boundary_term(n=nn, gamma=gamma, v_b=v_b, v_a=v_a, Nx=Nx, Nv=Nv, psi=psi)
        # res[nn * Nx: (nn + 1) * Nx] *= q / m * E
    return (res.reshape(Nv, Nx) * q / m * E).flatten()


def boundary_term(n, gamma, v_b, v_a, Nx, Nv, psi):
    """
    
    :param psi:
    :param Nv:
    :param Nx:
    :param v_a:
    :param v_b:
    :param gamma:
    :param n:
    :return: 
    """
    if n < 3:
        return 0
    else:
        return gamma / (v_b - v_a) * (
                xi_legendre(n=n, v=v_b, v_a=v_a, v_b=v_b) * construct_f(state=psi, v=v_b, Nv=Nv,
                                                                        Nx=Nx, v_a=v_a,
                                                                        v_b=v_b)
                - xi_legendre(n=n, v=v_a, v_a=v_a, v_b=v_b) * construct_f(state=psi, v=v_a, Nv=Nv, Nx=Nx, v_a=v_a,
                                                                          v_b=v_b))


def construct_f(state, v, Nv, Nx, v_a, v_b):
    """

    :param v_b:
    :param v_a:
    :param Nx:
    :param Nv:
    :param state:
    :param v:
    :return:
    """
    result = np.zeros(Nx)
    for n in range(Nv):
        result += state[n * Nx: (n + 1) * Nx] * xi_legendre(n=n, v=v, v_a=v_a, v_b=v_b)
    return result


def sigma_v2(n, i, v_a, v_b):
    """sigma(n, i)

    :param n: int, index of coefficients
    :param i: int, index of sum in nonlinear term
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :return: sigma(n, i)
    """
    # odd number
    if (n - i) % 2 == 1:
        return 2 * np.sqrt((2 * n + 1) * (2 * i + 1)) / (v_b - v_a)
    # even number
    else:
        return 0


def sigma_bar(v_a, v_b):
    """mean of v_a and v_b

    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :return: 0.5 * (v_a + v_b)
    """
    return 0.5 * (v_a + v_b)


def charge_density(q_e, q_i, C0_e, C0_i, v_a, v_b):
    """charge density (right hand side of Poisson equation)

    :param q_e: float, charge of electrons
    :param q_i: float, charge of ions
    :param C0_e: 1d array, density of electrons
    :param C0_i: 1d array, density of ions
    :param v_a: float, lower velocity boundary
    :param v_b: float, upper velocity boundary
    :return: change density rho(x, t=t*)
    """
    return (v_b - v_a) * (q_e * C0_e + q_i * C0_i)


def mass_legendre(state):
    """mass of a single specie

    :param state: 1d array, electron or ion state
    :return: mass for the state
    """
    return np.sum(state[0, :])


def momentum_legendre(state, v_a, v_b):
    """momentum of a single specie

    :param state: 1d array, electron or ion state
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :return: momentum for the state
    """
    return sigma_v1(n=1, v_a=v_a, v_b=v_b) * np.sum(state[1, :]) + 0.5 * (v_b + v_a) * np.sum(state[0, :])


def energy_k_legendre(state, v_a, v_b):
    """kinetic energy of a single specie

    :param state: 1d array, electron or ion state
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :return: kinetic energy for the state
    """
    return sigma_v1(n=1, v_a=v_a, v_b=v_b) * sigma_v1(n=2, v_a=v_a, v_b=v_b) * np.sum(state[2, :]) \
           + 2 * sigma_bar(v_a=v_a, v_b=v_b) * sigma_v1(n=1, v_a=v_a, v_b=v_b) * np.sum(state[1, :]) \
           + (sigma_v1(n=1, v_a=v_a, v_b=v_b)**2
              + sigma_v1(n=0, v_a=v_a, v_b=v_b)**2 + sigma_bar(v_a=v_a, v_b=v_b)**2) * np.sum(state[0, :])


def total_mass_legendre(state, v_a, v_b, dx):
    """total mass of single electron and ion setup

    :param state: 1d array, species s state
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :param dx: float, spatial spacing
    :return: total mass of single electron and ion setup
    """
    return mass_legendre(state=state) * dx * (v_b - v_a)


def total_momentum_legendre(state, v_a, v_b, dx, m_s):
    """total momentum of single electron and ion setup

    :param state: 1d array, species s state
    :param dx: float, spatial spacing
    :param m_s: float, mass of species s
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :return: total momentum of single electron and ion setup
    """
    return momentum_legendre(state=state, v_a=v_a, v_b=v_b) * dx * (v_b - v_a) * m_s


def total_energy_k_legendre(state, dx, v_a, v_b, m_s):
    """total kinetic energy of single electron and ion setup

    :param state: 1d array, species s state
    :param dx: float, spatial spacing
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :return: total kinetic energy of single electron and ion setup
    """
    return 0.5 * energy_k_legendre(state=state, v_a=v_a, v_b=v_b) * dx * (v_b - v_a) * m_s


def charge_density_two_stream_legendre(q_e1, q_e2, q_i, v_a, v_b, C0_e1, C0_e2, C0_i):
    """charge density (right hand side of Poisson equation)

    :param q_e1: float, charge of electrons species 1
    :param q_e2: float, charge of electrons species 2
    :param q_i: float, charge of ions
    :param C0_e1: 1d array, density of electrons species 1
    :param C0_e2: 1d array, density of electrons species 2
    :param C0_i: 1d array, density of ions
    :param v_a: float, the velocity lower boundary
    :param v_b: float, the velocity upper boundary
    :return: change density rho(x, t=t*)
    """
    return (v_b - v_a) * (q_e1 * C0_e1 + q_e2 * C0_e2 + q_i * C0_i)

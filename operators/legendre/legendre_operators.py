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
            A[n, n - 1] = sigma_v1(n, v_a, v_b)
        if n != Nv - 1:
            # upper diagonal
            A[n, n + 1] = sigma_v1(n + 1, v_a, v_b)
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
                res[nn * Nx: (nn + 1) * Nx] += psi[ii * Nx: (ii+1) * Nx] * sigma_v2(n=nn, i=ii, v_a=v_a, v_b=v_b)
        if gamma != 0:
            res[nn * Nx: (nn + 1) * Nx] += boundary_term(n=nn, gamma=gamma, v_b=v_b, v_a=v_a, Nx=Nx, Nv=Nv, psi=psi)
        res[nn * Nx: (nn + 1) * Nx] *= - q / m * E
    return res


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
    if n - i % 2 == 1:
        return 2 * np.sqrt((2 * n + 1) * (2 * i + 1)) / (v_b - v_a)
    # even number
    else:
        return 0


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

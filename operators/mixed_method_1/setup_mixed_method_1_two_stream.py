"""module to setup mixed method #1 with bulk Hermite and beam Legendre

ions are treated as stationary

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 27th, 2025
"""
import numpy as np
from operators.legendre.legendre_operators import A1_legendre, sigma_bar, B_legendre, xi_legendre
from operators.hermite.hermite_operators import A1_hermite, psi_hermite, psi_hermite_complement
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central
import scipy


class SimulationSetupMixedMethod1:
    def __init__(self, Nx, Nv_H, Nv_L, epsilon, v_a, v_b, alpha, u, gamma, L, dt, T0, T, nu_H, nu_L, Nv_int=int(1e4),
                 m_e=1, m_i=1836, q_e=-1, q_i=1, problem_dir=None, construct_integrals=True):
        # velocity grid
        # set up configuration parameters
        # spatial resolution
        self.Nx = Nx
        # velocity resolution
        self.Nv_H = Nv_H
        self.Nv_L = Nv_L
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity boundaries
        self.v_a = v_a
        self.v_b = v_b
        # hermite scaling and shifting parameters
        self.alpha = alpha
        self.u = u
        # penalty magnitude
        self.gamma = gamma
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping delta t
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e = m_e
        self.m_i = m_i
        # charge normalized
        self.q_e = q_e
        self.q_i = q_i
        # artificial collisional frequency
        self.nu_H = nu_H
        self.nu_L = nu_L
        # directory name
        self.problem_dir = problem_dir
        self.construct_integrals = construct_integrals

        # matrices
        # finite difference derivative matrix
        self.D = ddx_central(Nx=self.Nx + 1, dx=self.dx, periodic=True, order=2)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # Hermite operator
        self.A_e_H = self.alpha * A1_hermite(D=self.D, Nv=self.Nv_H) \
                     + self.u * A2(D=self.D, Nv=self.Nv_H) \
                     + self.nu_H * A3(Nx=self.Nx, Nv=self.Nv_H)

        # Legendre operators
        self.A_e_L = A1_legendre(D=self.D, Nv=self.Nv_L, v_a=v_a, v_b=v_b) \
                     + sigma_bar(v_a=self.v_a, v_b=self.v_b) * A2(D=self.D, Nv=self.Nv_L) \
                     + self.nu_L * A3(Nx=self.Nx, Nv=self.Nv_L)

        self.B_e_L = B_legendre(Nv=self.Nv_L, Nx=self.Nx, v_a=self.v_a, v_b=self.v_b)

        # xi functions
        self.xi_v_a = np.zeros(self.Nv_L)
        self.xi_v_b = np.zeros(self.Nv_L)
        for nn in range(self.Nv_L):
            self.xi_v_a[nn] = xi_legendre(n=nn, v=self.v_a, v_a=self.v_a, v_b=self.v_b)
            self.xi_v_b[nn] = xi_legendre(n=nn, v=self.v_b, v_a=self.v_a, v_b=self.v_b)

        if construct_integrals:
            v_ = np.linspace(v_a, v_b, Nv_int, endpoint=True)

            self.J_int = np.zeros((self.Nv_H + 1, self.Nv_L))
            self.I_int_complement = np.zeros((self.Nv_H + 1, self.Nv_L))

            for mm in range(self.Nv_L):
                for nn in range(self.Nv_H + 1):
                    if (mm % 2 == 0) and (nn % 2 == 1) and self.v_a == -self.v_b:
                        self.J_int[nn, mm] = 0
                        self.I_int_complement[nn, mm] = 0
                    elif (mm % 2 == 1) and (nn % 2 == 0) and self.v_a == -self.v_b:
                        self.J_int[nn, mm] = 0
                        self.I_int_complement[nn, mm] = 0
                    else:
                        self.J_int[nn, mm] = scipy.integrate.trapezoid(
                                xi_legendre(n=mm, v=v_, v_a=self.v_a, v_b=self.v_b)
                                * psi_hermite(n=nn, alpha_s=self.alpha, u_s=self.u, v=v_),
                                x=v_, dx=np.abs(v_[1] - v_[0]))

                        self.I_int_complement[nn, mm] = scipy.integrate.trapezoid(
                            xi_legendre(n=mm, v=v_, v_a=self.v_a, v_b=self.v_b)
                            * psi_hermite_complement(n=nn, alpha_s=self.alpha, u_s=self.u, v=v_),
                            x=v_, dx=np.abs(v_[1] - v_[0]))

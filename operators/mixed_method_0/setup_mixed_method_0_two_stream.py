"""module to setup mixed (static) method with bulk Hermite and beam Legendre

ions are treated as stationary

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 9th, 2025
"""

import numpy as np
from operators.legendre.legendre_operators import A1_legendre, sigma_bar, B_legendre, xi_legendre
from operators.hermite.hermite_operators import A1_hermite
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central


class SimulationSetupMixedMethod0:
    def __init__(self, Nx, Nv_e1, Nv_e2, epsilon, v_a, v_b, alpha_e1, u_e1, u_e2, gamma, L, dt, T0, T, nu_H, nu_L, n0_e1, n0_e2,
                 u_tol, alpha_tol, m_e=1, m_i=1836, q_e=-1, q_i=1, problem_dir=None):
        # velocity grid
        # set up configuration parameters
        # spatial resolution
        self.Nx = Nx
        # velocity resolution
        self.Nv_e1 = Nv_e1
        self.Nv_e2 = Nv_e2
        # average density coefficient
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
        # parameters tolerances
        self.u_tol = u_tol
        self.alpha_tol = alpha_tol
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity boundaries
        self.v_a = v_a
        self.v_b = v_b
        # hermite scaling and shifting parameters
        self.alpha_e1 = [alpha_e1]
        self.u_e1 = [u_e1]
        self.u_e2 = u_e2
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

        # matrices
        # finite difference derivative matrix
        self.D = ddx_central(Nx=self.Nx + 1, dx=self.dx, periodic=True, order=2)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # Hermite operator
        self.A_eH_diag = A2(D=self.D, Nv=self.Nv_e1)
        self.A_eH_off = A1_hermite(D=self.D, Nv=self.Nv_e1)
        self.A_eH_col = A3(Nx=self.Nx, Nv=self.Nv_e1)

        # Legendre operators
        self.A_e_L = A1_legendre(D=self.D, Nv=self.Nv_e2, v_a=v_a, v_b=v_b) \
                     + sigma_bar(v_a=self.v_a, v_b=self.v_b) * A2(D=self.D, Nv=self.Nv_e2) \
                     + self.nu_L * A3(Nx=self.Nx, Nv=self.Nv_e2)

        self.B_e_L = B_legendre(Nv=self.Nv_e2, Nx=self.Nx, v_a=self.v_a, v_b=self.v_b)

        # xi functions
        self.xi_v_a = np.zeros(self.Nv_e2)
        self.xi_v_b = np.zeros(self.Nv_e2)
        for nn in range(self.Nv_e2):
            self.xi_v_a[nn] = xi_legendre(n=nn, v=self.v_a, v_a=self.v_a, v_b=self.v_b)
            self.xi_v_b[nn] = xi_legendre(n=nn, v=self.v_b, v_a=self.v_a, v_b=self.v_b)

    def add_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1.append(alpha_e1_curr)

    def add_u_e1(self, u_e1_curr):
        self.u_e1.append(u_e1_curr)

    def replace_alpha_e1(self, alpha_e1_curr):
        self.alpha_e1[-1] = alpha_e1_curr

    def replace_u_e1(self, u_e1_curr):
        self.u_e1[-1] = u_e1_curr

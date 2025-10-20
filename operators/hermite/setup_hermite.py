"""module to setup Hermite simulation.

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np
from operators.hermite.hermite_operators import A1_hermite
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central


class SimulationSetupHermite:
    def __init__(self, Nx, Nv, epsilon, alpha_e, alpha_i, u_e, u_i, L, dt, T0, T, nu,
                 m_e=1, m_i=1836, q_e=-1, q_i=1, problem_dir=None):
        # set up configuration parameters
        # spatial resolution
        self.Nx = Nx
        # velocity resolution
        self.Nv = Nv
        # total number of DOF for each species
        self.N = self.Nx * self.Nv
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # in the adaptive setting alpha and u are vectors and are tracked in time.
        # velocity scaling parameter (mean thermal velocity)
        self.alpha_e = [alpha_e]
        self.alpha_i = [alpha_i]
        # velocity shifting parameter (mean fluid velocity)
        self.u_e = [u_e]
        self.u_i = [u_i]
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
        self.nu = nu
        # directory name
        self.problem_dir = problem_dir

        # matrices
        # finite difference derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=True, order=2)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # matrix of coefficients (advection)
        self.A_diag = A2(D=self.D, Nv=self.Nv)
        self.A_off = A1_hermite(D=self.D, Nv=self.Nv)
        self.A_col = A3(Nx=self.Nx, Nv=self.Nv)

    def add_alpha_e(self, alpha_e_curr):
        self.alpha_e.append(alpha_e_curr)

    def add_alpha_i(self, alpha_i_curr):
        self.alpha_i.append(alpha_i_curr)

    def add_u_e(self, u_e_curr):
        self.u_e.append(u_e_curr)

    def add_u_i(self, u_i_curr):
        self.u_i.append(u_i_curr)

    def replace_alpha_e(self, alpha_e_curr):
        self.alpha_e[-1] = alpha_e_curr

    def replace_alpha_i(self, alpha_i_curr):
        self.alpha_i[-1] = alpha_i_curr

    def replace_u_e(self, u_e_curr):
        self.u_e[-1] = u_e_curr

    def replace_u_i(self, u_i_curr):
        self.u_i[-1] = u_i_curr

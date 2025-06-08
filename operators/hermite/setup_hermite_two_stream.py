"""module to setup Hermite simulation with two electron populations

Author: Opal Issan (oissan@ucsd.edu)
Last Update: June 6th, 2025
"""
import numpy as np
from operators.hermite.hermite_operators import A1
from operators.universal_functions import get_D_inv, A2, A3
from operators.finite_difference import ddx_central


class SimulationSetupTwoStreamHermite:
    def __init__(self, Nx,  Nv_e1, Nv_e2, epsilon, alpha_e1, alpha_e2, alpha_i,
                 u_e1, u_e2, u_i, L, dt, T0, T,
                 nu_e1, nu_e2,  n0_e1, n0_e2, FD_order,
                 periodic=True, nu_i=0, m_e1=1, m_e2=1, m_i=1836, q_e1=-1, q_e2=-1, q_i=1, ions=False, Nv_i=0):
        # set up configuration parameters
        # resolution in space
        self.Nx = Nx
        # resolution in velocity
        self.Nv_e1 = Nv_e1
        self.Nv_e2 = Nv_e2
        self.Nv_i = Nv_i
        # total DOF for each species
        # self.N = self.Nv * self.Nx
        # epsilon displacement in initial electron distribution
        self.epsilon = epsilon
        # velocity scaling of electron and ion
        self.alpha_e1 = alpha_e1
        self.alpha_e2 = alpha_e2
        self.alpha_i = alpha_i
        # velocity scaling
        self.u_e1 = u_e1
        self.u_e2 = u_e2
        self.u_i = u_i
        # average density coefficient
        self.n0_e1 = n0_e1
        self.n0_e2 = n0_e2
        # x grid is from 0 to L
        self.L = L
        self.dx = self.L / self.Nx
        # time stepping
        self.dt = dt
        # final time
        self.T = T
        # initial start
        self.T0 = T0
        # vector of timestamps
        self.t_vec = np.linspace(self.T0, self.T, int((self.T - self.T0) / self.dt) + 1)
        # mass normalized
        self.m_e1 = m_e1
        self.m_e2 = m_e2
        self.m_i = m_i
        # charge normalized
        self.q_e1 = q_e1
        self.q_e2 = q_e2
        self.q_i = q_i
        # artificial collisional frequency
        self.nu_e1 = nu_e1
        self.nu_e2 = nu_e2
        self.nu_i = nu_i
        # order of finite difference operator
        self.FD_order = FD_order

        # matrices
        # Fourier derivative matrix
        self.D = ddx_central(Nx=self.Nx+1, dx=self.dx, periodic=periodic, order=FD_order)
        self.D_inv = get_D_inv(Nx=self.Nx, D=self.D)

        # A matrices
        self.A_e1 = self.u_e1 * A2(D=self.D, Nv=self.Nv_e1) \
                    + self.alpha_e1 * A1(D=self.D, Nv=self.Nv_e1) \
                    + self.nu_e1 * A3(Nx=self.Nx, Nv=self.Nv_e1)
        self.A_e2 = self.u_e2 * A2(D=self.D, Nv=self.Nv_e2) \
                    + self.alpha_e2 * A1(D=self.D, Nv=self.Nv_e2) \
                    + self.nu_e2 * A3(Nx=self.Nx, Nv=self.Nv_e2)

        # if ions evolve
        if ions:
            self.A_i = self.u_i * A2(D=self.D, Nv=self.Nv_i) \
                       + self.alpha_i * A1(D=self.D, Nv=self.Nv_i) \
                       + self.nu_i * A3(Nx=self.Nx, Nv=self.Nv_i)

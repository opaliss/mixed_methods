"""Module to run mixed method #0 (static) bump on tail testcase

Author: Opal Issan
Date: June 9th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_0.mixed_method_0_operators import charge_density_two_stream_mixed_method_0
from operators.legendre.legendre_operators import nonlinear_legendre, xi_legendre
from operators.hermite.hermite_operators import nonlinear_hermite
from operators.mixed_method_0.setup_mixed_method_0_two_stream import SimulationSetupMixedMethod0
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import scipy


def rhs(y):
    # charge density computed
    rho = charge_density_two_stream_mixed_method_0(q_e=setup.q_e, alpha_e=setup.alpha, v_a=setup.v_a, v_b=setup.v_b,
                                                   C0_e_hermite=y[:setup.Nx],
                                                   C0_e_legendre=y[setup.Nv_H * setup.Nx: (setup.Nv_H + 1) * setup.Nx])

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    dydt_ = np.zeros(len(y))

    # evolving bulk hermite
    dydt_[:setup.Nv_H * setup.Nx] = setup.A_e_H @ y[:setup.Nv_H * setup.Nx] + nonlinear_hermite(E=E, psi=y[:setup.Nv_H * setup.Nx],
                                                                                                q=setup.q_e,
                                                                                                m=setup.m_e,
                                                                                                alpha=setup.alpha,
                                                                                                Nv=setup.Nv_H,
                                                                                                Nx=setup.Nx)

    dydt_[setup.Nv_H * setup.Nx:] = setup.A_e_L @ y[setup.Nv_H * setup.Nx:] + nonlinear_legendre(E=E, psi=y[setup.Nv_H * setup.Nx:],
                                                                                                 Nv=setup.Nv_L,
                                                                                                 Nx=setup.Nx,
                                                                                                 B_mat=setup.B_e_L,
                                                                                                 q=setup.q_e,
                                                                                                 m=setup.m_e,
                                                                                                 gamma=setup.gamma,
                                                                                                 v_a=setup.v_a,
                                                                                                 v_b=setup.v_b,
                                                                                                 xi_v_a=setup.xi_v_a,
                                                                                                 xi_v_b=setup.xi_v_b)

    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod0(Nx=51,
                                        Nv_H=50,
                                        Nv_L=100,
                                        epsilon=1e-2,
                                        v_a=-7,
                                        v_b=7,
                                        alpha=np.sqrt(2),
                                        u=0,
                                        L=20 * np.pi / 3,
                                        dt=1e-2,
                                        T0=0,
                                        T=20,
                                        nu_L=1,
                                        nu_H=10,
                                        gamma=0.5)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_H + setup.Nv_L) * setup.Nx)
    # bulk electrons => hermite
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = 0.9 * (np.ones(setup.Nx) + setup.epsilon * np.cos(0.3 * x_)) / setup.alpha
    # beam electrons => legendre
    v_ = np.linspace(setup.v_a, setup.v_b, 1000, endpoint=True)
    x_component = (1 + setup.epsilon * np.cos(0.3 * x_)) / (setup.v_b - setup.v_a) / np.sqrt(np.pi)
    for nn in range(setup.Nv_L):
        xi = xi_legendre(n=nn, v=v_, v_a=setup.v_a, v_b=setup.v_b)
        exp_ = 0.1 * np.exp(-2 * ((v_ - 4.5) ** 2)) * np.sqrt(2)
        v_component = scipy.integrate.trapezoid(xi * exp_, x=v_, dx=np.abs(v_[1] - v_[0]))
        y0[setup.Nx*setup.Nv_H + nn * setup.Nx: setup.Nx*setup.Nv_H + (nn + 1) * setup.Nx] = x_component * v_component

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(y_0=y0,
                                              right_hand_side=rhs,
                                              a_tol=1e-10,
                                              r_tol=1e-10,
                                              max_iter=100,
                                              param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save("../../data/mixed_method_0/bump_on_tail/sol_runtime_NvH_" + str(setup.Nv_H) + "_NvL_" + str(setup.Nv_L) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/mixed_method_0/bump_on_tail/sol_u_NvH_" + str(setup.Nv_H) + "_NvL_" + str(setup.Nv_L) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/mixed_method_0/bump_on_tail/sol_t_NvH_" + str(setup.Nv_H) + "_NvL_" + str(setup.Nv_L) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

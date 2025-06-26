"""Module to run mixed method #1 nonlinear landau testcase

Author: Opal Issan
Date: June 11th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_0.mixed_method_0_operators import charge_density_two_stream_mixed_method_0
from operators.mixed_method_1.mixed_method_1_operators import extra_term_1
from operators.legendre.legendre_operators import nonlinear_legendre
from operators.hermite.hermite_operators import nonlinear_hermite
from operators.mixed_method_1.setup_mixed_method_1_two_stream import SimulationSetupMixedMethod1
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed
    rho = charge_density_two_stream_mixed_method_0(q_e=setup.q_e, alpha_e=setup.alpha,
                                                   v_a=setup.v_a, v_b=setup.v_b,
                                                   C0_e_hermite=y[:setup.Nx],
                                                   C0_e_legendre=y[setup.Nv_H * setup.Nx: (setup.Nv_H + 1) * setup.Nx])

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    dydt_ = np.zeros(len(y))

    # evolving bulk hermite
    dydt_[:setup.Nv_H * setup.Nx] = setup.A_e_H @ y[:setup.Nv_H * setup.Nx] \
                                    + nonlinear_hermite(E=E,
                                                        psi=y[:setup.Nv_H * setup.Nx],
                                                        q=setup.q_e,
                                                        m=setup.m_e,
                                                        alpha=setup.alpha,
                                                        Nv=setup.Nv_H,
                                                        Nx=setup.Nx)

    dydt_[setup.Nv_H * setup.Nx:] = setup.A_e_L @ y[setup.Nv_H * setup.Nx:] \
                                    + nonlinear_legendre(E=E, psi=y[setup.Nv_H * setup.Nx:],
                                                         Nv=setup.Nv_L,
                                                         Nx=setup.Nx,
                                                         B_mat=setup.B_e_L,
                                                         q=setup.q_e,
                                                         m=setup.m_e,
                                                         gamma=setup.gamma,
                                                         v_a=setup.v_a,
                                                         v_b=setup.v_b,
                                                         xi_v_a=setup.xi_v_a,
                                                         xi_v_b=setup.xi_v_b) \
                                    + extra_term_1(LH_int_1=setup.LH_int[:, -1],
                                                   v_b=setup.v_b,
                                                   v_a=setup.v_a,
                                                   C_hermite_last=y[(setup.Nv_H - 1) * setup.Nx: setup.Nv_H * setup.Nx],
                                                   alpha=setup.alpha,
                                                   Nv_H=setup.Nv_H,
                                                   D=setup.D,
                                                   E=E,
                                                   Nv_L=setup.Nv_L,
                                                   Nx=setup.Nx)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod1(Nx=101,
                                        Nv_H=100,
                                        Nv_L=100,
                                        epsilon=1e-2,
                                        v_a=-10,
                                        v_b=10,
                                        alpha=np.sqrt(2),
                                        u=0,
                                        L=2 * np.pi,
                                        dt=1e-2,
                                        T0=0,
                                        T=100,
                                        nu_L=0,
                                        nu_H=0,
                                        gamma=0.5,
                                        construct_integrals=True)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_H + setup.Nv_L) * setup.Nx)
    # grid
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    # initial condition (only initialize Hermite zeroth coefficient)
    y0[:setup.Nx] = (1 + setup.epsilon * np.cos(x_)) / setup.alpha

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
    np.save(
        "../../data/mixed_method_1_hermite_legendre/weak_landau/sol_runtime_NvH_" + str(setup.Nv_H) + "_NvL_" + str(
            setup.Nv_L) +
        "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/mixed_method_1_hermite_legendre/weak_landau/sol_u_NvH_" + str(setup.Nv_H) + "_NvL_" + str(
        setup.Nv_L) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/mixed_method_1_hermite_legendre/weak_landau/sol_t_NvH_" + str(setup.Nv_H) + "_NvL_" + str(
        setup.Nv_L) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

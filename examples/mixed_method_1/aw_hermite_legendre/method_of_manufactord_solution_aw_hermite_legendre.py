"""Module to run mixed method #1 bump on tail testcase

Author: Opal Issan
Date: June 11th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.mixed_method_0.mixed_method_0_operators import charge_density_two_stream_mixed_method_0
from operators.mixed_method_1.mixed_method_1_operators import extra_term_1, closure_term
from operators.legendre.legendre_operators import nonlinear_legendre, xi_legendre
from operators.aw_hermite.aw_hermite_operators import nonlinear_aw_hermite, aw_psi_hermite_complement
from operators.mixed_method_1.setup_mixed_method_1_two_stream import SimulationSetupMixedMethod1
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import scipy


def rhs(y):
    # electric field computed (poisson solver)
    E = np.zeros(setup.Nx)

    dydt_ = np.zeros(len(y))

    # evolving bulk aw_hermite
    dydt_[:setup.Nv_e1 * setup.Nx] = setup.A_e_H @ y[:setup.Nv_e1 * setup.Nx] \
                                     + nonlinear_aw_hermite(E=E,
                                                            psi=y[:setup.Nv_e1 * setup.Nx],
                                                            q=setup.q_e,
                                                            m=setup.m_e,
                                                            alpha=setup.alpha_e1,
                                                            Nv=setup.Nv_e1,
                                                            Nx=setup.Nx)

    dydt_[setup.Nv_e1 * setup.Nx:] = setup.A_e_L @ y[setup.Nv_e1 * setup.Nx:] \
                                     + nonlinear_legendre(E=E, psi=y[setup.Nv_e1 * setup.Nx:],
                                                          Nv=setup.Nv_e2,
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
                                                    C_hermite_last=y[(setup.Nv_e1 - 1) * setup.Nx: setup.Nv_e1 * setup.Nx],
                                                    alpha=setup.alpha_e1,
                                                    Nv_H=setup.Nv_e1,
                                                    D=setup.D,
                                                    E=E,
                                                    Nx=setup.Nx,
                                                    Nv_L=setup.Nv_e2)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupMixedMethod1(Nx=101,
                                        Nv_e1=80,
                                        Nv_e2=80,
                                        epsilon=1e-2,
                                        v_a=-5,
                                        v_b=5,
                                        alpha_e1=np.sqrt(2),
                                        u_e1=0,
                                        L=20 * np.pi / 3,
                                        dt=1e-2,
                                        T0=0,
                                        T=80,
                                        nu_L=0,
                                        nu_H=0,
                                        gamma=0.5,
                                        construct_integrals=True)

    # initial condition: read in result from previous simulation
    y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
    # grid
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = (1 + np.cos(0.3 * x_)) / setup.alpha_e1

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
        "../../data/mixed_method_1_aw_hermite_legendre/manufactored_solution/sol_runtime_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
            setup.Nv_e2) +
        "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/mixed_method_1_aw_hermite_legendre/manufactored_solution/sol_u_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/mixed_method_1_aw_hermite_legendre/manufactored_solution/sol_t_NvH_" + str(setup.Nv_e1) + "_NvL_" + str(
        setup.Nv_e2) +
            "_Nx_" + str(setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

"""Module to run the bump-on-tail instability Hermite testcase

Author: Opal Issan (oissan@ucsd.edu)
Last modified: Oct 22nd, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.hermite.hermite_operators import nonlinear_hermite, charge_density_two_stream_hermite
from operators.implicit_midpoint_adaptive_two_stream import implicit_midpoint_solver_adaptive_two_stream
from operators.hermite.setup_hermite_two_stream import SimulationSetupTwoStreamHermite
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed for poisson's equation
    rho = charge_density_two_stream_hermite(C0_e1=y[:setup.Nx],
                                            C0_e2=y[setup.Nx * setup.Nv_e1: setup.Nx * (setup.Nv_e1 + 1)],
                                            C0_i=np.ones(setup.Nx) / setup.alpha_i[-1],
                                            alpha_e1=setup.alpha_e1[-1],
                                            alpha_e2=setup.alpha_e2[-1],
                                            alpha_i=setup.alpha_i[-1],
                                            q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i)

    # electric field computed
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # initialize the rhs dydt
    dydt_ = np.zeros(len(y))
    # evolving electrons
    # electron species (1) => bulk
    A_e1 = setup.u_e1[-1] * setup.A_diag_e1 + setup.alpha_e1[-1] * setup.A_off_e1 + setup.nu_e1 * setup.A_col_e1
    A_e2 = setup.u_e2[-1] * setup.A_diag_e2 + setup.alpha_e2[-1] * setup.A_off_e2 + setup.nu_e2 * setup.A_col_e2

    dydt_[:setup.Nv_e1 * setup.Nx] = A_e1 @ y[:setup.Nv_e1 * setup.Nx] \
                                     + nonlinear_hermite(E=E, psi=y[:setup.Nv_e1 * setup.Nx], Nv=setup.Nv_e1,
                                                         Nx=setup.Nx,
                                                         q=setup.q_e1, m=setup.m_e1, alpha=setup.alpha_e1[-1])

    # electron species (2) => bump
    dydt_[setup.Nv_e1 * setup.Nx:] = A_e2 @ y[setup.Nv_e1 * setup.Nx:] \
                                     + nonlinear_hermite(E=E, psi=y[setup.Nv_e1 * setup.Nx:], Nv=setup.Nv_e2,
                                                         Nx=setup.Nx,
                                                         q=setup.q_e2, m=setup.m_e2, alpha=setup.alpha_e2[-1])
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupTwoStreamHermite(Nx=101,
                                            Nv_e1=100,
                                            Nv_e2=100,
                                            epsilon=1e-2,
                                            alpha_e1=np.sqrt(2),
                                            alpha_e2=1 / np.sqrt(2),
                                            alpha_i=np.sqrt(2 / 1836),
                                            u_e1=0,
                                            u_e2=4.5,
                                            u_i=0,
                                            L=20 * np.pi / 3,
                                            dt=1e-2,
                                            T0=0,
                                            T=40,
                                            k0=1,
                                            nu_e1=4,
                                            nu_e2=4,
                                            n0_e1=0.9,
                                            n0_e2=0.1,
                                            alpha_tol=1e-1,
                                            u_tol=1e-1)

    # initial condition: read in result from previous simulation
    # ions (unperturbed + static)
    y0 = np.zeros((setup.Nv_e1 + setup.Nv_e2) * setup.Nx)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    # first electron species ==> "bulk" (perturbed)
    y0[:setup.Nx] = setup.n0_e1 * (1 + setup.epsilon * np.cos(x_ * setup.k0 / setup.L * 2 * np.pi)) / setup.alpha_e1[-1]
    # second electron species ==> "bump" (perturbed)
    y0[setup.Nv_e1 * setup.Nx: setup.Nv_e1 * setup.Nx + setup.Nx] = setup.n0_e2 / setup.alpha_e2[-1]

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver_adaptive_two_stream(y_0=y0,
                                                                         right_hand_side=rhs,
                                                                         r_tol=1e-10,
                                                                         a_tol=1e-10,
                                                                         max_iter=100,
                                                                         bump_hermite_adapt=True,
                                                                         bulk_hermite_adapt=True,
                                                                         adaptive=True,
                                                                         param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save runtime
    np.save("../../data/hermite/bump_on_tail_adaptive/sol_runtime_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx) + "_" + str(setup.T0)
            + "_" + str(setup.T) + ".npy", np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/hermite/bump_on_tail_adaptive/sol_u_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", sol_midpoint_u)

    np.save("../../data/hermite/bump_on_tail_adaptive/sol_t_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.t_vec)

    # save time varying alpha and u
    np.save("../../data/hermite/bump_on_tail_adaptive/alpha_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e1)

    np.save("../../data/hermite/bump_on_tail_adaptive/alpha_e2_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.alpha_e2)

    np.save("../../data/hermite/bump_on_tail_adaptive/u_e1_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e1)

    np.save("../../data/hermite/bump_on_tail_adaptive/u_e2_Nve1_" + str(setup.Nv_e1)
            + "_Nve2_" + str(setup.Nv_e2) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T) + ".npy", setup.u_e2)

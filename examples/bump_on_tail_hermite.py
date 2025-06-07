"""Module to run the bump-on-tail instability full-order model (FOM) testcase

Author: Opal Issan
Date: June 6th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.hermite.hermite_operators import nonlinear_full, charge_density_two_stream
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.hermite.setup_hermite_two_stream import SimulationSetupTwoStreamHermite
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed for poisson's equation
    rho = charge_density_two_stream(C0_e1=y[:setup.Nx],
                                    C0_e2=y[setup.Nx * setup.Nv: setup.Nx * (setup.Nv + 1)],
                                    C0_i=C0_ions, alpha_e1=setup.alpha_e1, alpha_e2=setup.alpha_e2,
                                    alpha_i=setup.alpha_i, q_e1=setup.q_e1, q_e2=setup.q_e2, q_i=setup.q_i)

    # electric field computed
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # initialize the rhs dydt
    dydt_ = np.zeros(len(y))
    # evolving electrons
    # electron species (1) => bulk
    dydt_[:setup.Nv * setup.Nx] = setup.A_e1 @ y[:setup.Nv * setup.Nx] \
                                  + nonlinear_full(E=E, psi=y[:setup.Nv * setup.Nx], Nv=setup.Nv, Nx=setup.Nx,
                                                q=setup.q_e1, m=setup.m_e1, alpha=setup.alpha_e1)

    # electron species (2) => bump
    dydt_[setup.Nv * setup.Nx:] = setup.A_e2 @ y[setup.Nv * setup.Nx:] \
                                  + nonlinear_full(E=E, psi=y[setup.Nv * setup.Nx:], Nv=setup.Nv, Nx=setup.Nx,
                                                   q=setup.q_e2, m=setup.m_e2, alpha=setup.alpha_e2)
    return dydt_


if __name__ == "__main__":
    setup = SimulationSetupTwoStreamHermite(Nx=301,
                                            Nv=200,
                                            epsilon=1e-3,
                                            alpha_e1=np.sqrt(2),
                                            alpha_e2=1/np.sqrt(2),
                                            alpha_i=np.sqrt(2 / 1836),
                                            u_e1=0,
                                            u_e2=4.5,
                                            u_i=0,
                                            L=20 * np.pi / 3,
                                            dt=1e-2,
                                            T0=0,
                                            T=80,
                                            nu_e1=20,
                                            nu_e2=20,
                                            n0_e1=0.9,
                                            n0_e2=0.1,
                                            FD_order=2)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(2 * setup.Nv * setup.Nx)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = setup.n0_e1 * (np.ones(setup.Nx) + setup.epsilon * np.cos(0.3 * x_)) / setup.alpha_e1
    # second electron species (unperturbed)
    y0[setup.Nv * setup.Nx: setup.Nv * setup.Nx + setup.Nx] = setup.n0_e2 * np.ones(setup.Nx) / setup.alpha_e2
    # ions (unperturbed + static)
    C0_ions = np.ones(setup.Nx) / setup.alpha_i

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u, setup = implicit_midpoint_solver(y_0=y0,
                                                     right_hand_side=rhs,
                                                     r_tol=1e-6,
                                                     a_tol=1e-8,
                                                     max_iter=100,
                                                     param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save runtime
    np.save("../data/hermite/bump_on_tail/sol_runtime_Nv_" + str(setup.Nv) + "_Nx_" + str(
        setup.Nx) + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../data/hermite/bump_on_tail/sol_u_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)
    np.save("../data/hermite/bump_on_tail/sol_t_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)


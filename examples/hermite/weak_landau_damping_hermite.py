"""Module to run Hermite weak_landau Landau damping testcase

Author: Opal Issan
Date: June 6, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.hermite.hermite_operators import nonlinear_full, charge_density
from operators.hermite.setup_hermite import SimulationSetupHermite
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.poisson_solver import gmres_solver
import time
import numpy as np


def rhs(y):
    # charge density computed
    rho = charge_density(alpha_e=setup.alpha_e, alpha_i=setup.alpha_i, q_e=setup.q_e, q_i=setup.q_i,
                         C0_e=y[:setup.Nx], C0_i=np.ones(setup.Nx) / setup.alpha_i)

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # evolving only electrons
    return setup.A_e @ y + nonlinear_full(E=E, psi=y, Nv=setup.Nv, Nx=setup.Nx, alpha=setup.alpha_e,
                                          q=setup.q_e, m=setup.m_e)


if __name__ == "__main__":
    setup = SimulationSetupHermite(Nx=51,
                                   Nv=50,
                                   epsilon=1e-2,
                                   alpha_e=0.75,
                                   alpha_i=np.sqrt(2 / 1836),
                                   u_e=0,
                                   u_i=0,
                                   L=2 * np.pi,
                                   dt=1e-2,
                                   T0=0,
                                   T=10,
                                   nu=5)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.Nv * setup.Nx)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    y0[:setup.Nx] = (1 + setup.epsilon * np.cos(x_)) / setup.alpha_e

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(y_0=y0,
                                              right_hand_side=rhs,
                                              a_tol=1e-9,
                                              r_tol=1e-9,
                                              max_iter=100,
                                              param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save("../data/hermite/weak_landau/sol_runtime_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../data/hermite/weak_landau/sol_u_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx) + "_"
            + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../data/hermite/weak_landau/sol_t_Nv_" + str(setup.Nv) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

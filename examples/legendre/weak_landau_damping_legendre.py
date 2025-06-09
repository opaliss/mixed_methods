"""Module to run Legendre weak Landau damping testcase

Author: Opal Issan
Date: June 9th, 2025
"""
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

from operators.legendre.legendre_operators import nonlinear_legendre, charge_density_legendre, xi_legendre
from operators.legendre.setup_legendre import SimulationSetupLegendre
from operators.implicit_midpoint import implicit_midpoint_solver
from operators.poisson_solver import gmres_solver
import time
import numpy as np
import scipy


def rhs(y):
    # charge density computed
    rho = charge_density_legendre(q_e=setup.q_e, q_i=setup.q_i, C0_e=y[:setup.Nx],
                                  C0_i=np.ones(setup.Nx)/(setup.v_b - setup.v_a),
                                  v_a=setup.v_a, v_b=setup.v_b)

    # electric field computed (poisson solver)
    E = gmres_solver(rhs=rho, D=setup.D, D_inv=setup.D_inv, a_tol=1e-12, r_tol=1e-12)

    # evolving only electrons
    return setup.A_e @ y + nonlinear_legendre(E=E, psi=y, Nv=setup.Nv_e, Nx=setup.Nx,
                                              q=setup.q_e, m=setup.m_e, gamma=setup.gamma,
                                              v_a=setup.v_a, v_b=setup.v_b, xi_v_a=setup.xi_v_a,
                                              xi_v_b=setup.xi_v_b, B_mat=setup.B_e)


if __name__ == "__main__":
    setup = SimulationSetupLegendre(Nx=21,
                                    Nv_e=50,
                                    epsilon=1e-2,
                                    v_a=-5,
                                    v_b=5,
                                    gamma=0,
                                    L=2 * np.pi,
                                    dt=1e-2,
                                    T0=0,
                                    T=10,
                                    nu=0)

    # initial condition: read in result from previous simulation
    y0 = np.zeros(setup.Nv_e * setup.Nx)
    # first electron 1 species (perturbed)
    x_ = np.linspace(0, setup.L, setup.Nx, endpoint=False)
    v_ = np.linspace(setup.v_a, setup.v_b, 1000, endpoint=True)
    x_component = (1 + setup.epsilon * np.cos(x_)) / np.sqrt(2 * np.pi) / (setup.v_b - setup.v_a)
    for nn in range(setup.Nv_e):
        xi = xi_legendre(n=nn, v=v_, v_a=setup.v_a, v_b=setup.v_b)
        v_component = scipy.integrate.trapezoid(xi * np.exp(-0.5 * (v_ ** 2)), x=v_, dx=np.abs(v_[1] - v_[0]))
        y0[nn * setup.Nx: (nn + 1) * setup.Nx] = x_component * v_component

    # start timer
    start_time_cpu = time.process_time()
    start_time_wall = time.time()

    # integrate (implicit midpoint)
    sol_midpoint_u = implicit_midpoint_solver(y_0=y0,
                                              right_hand_side=rhs,
                                              a_tol=1e-8,
                                              r_tol=1e-8,
                                              max_iter=100,
                                              param=setup)

    end_time_cpu = time.process_time() - start_time_cpu
    end_time_wall = time.time() - start_time_wall

    print("runtime cpu = ", end_time_cpu)
    print("runtime wall = ", end_time_wall)

    # save the runtime
    np.save("../../data/legendre/weak_landau/sol_runtime_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), np.array([end_time_cpu, end_time_wall]))

    # save results
    np.save("../../data/legendre/weak_landau/sol_u_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx) + "_"
            + str(setup.T0) + "_" + str(setup.T), sol_midpoint_u)

    np.save("../../data/legendre/weak_landau/sol_t_Nv_" + str(setup.Nv_e) + "_Nx_" + str(setup.Nx)
            + "_" + str(setup.T0) + "_" + str(setup.T), setup.t_vec)

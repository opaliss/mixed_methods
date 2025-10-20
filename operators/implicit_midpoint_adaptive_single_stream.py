"""Module includes temporal integrator and adaptivity

Authors: Opal Issan (oissan@ucsd.edu)
Version: Oct 20th, 2025
"""
import numpy as np
import scipy
from operators.adaptive_hermite import check_if_update_needed, updated_u, updated_alpha, get_projection_matrix
from operators.implicit_midpoint_adaptive_two_stream import implicit_nonlinear_equation


def implicit_midpoint_solver_adaptive_single_stream(y_0, right_hand_side, param, r_tol=1e-8, a_tol=1e-15, max_iter=100,
                                                    adaptive=True):
    """Solve the system

        dy/dt = rhs(y),    y(0) = y0,

    via the implicit midpoint method.

    The nonlinear equation at each time step is solved using Anderson acceleration.

    Parameters
    ----------
    :param bulk_hermite:
    :param bump_hermite:
    :param param: object of SimulationSetup with all the simulation setup parameters
    :param max_iter: maximum iterations of nonlinear solver, default is 100
    :param a_tol: absolute tolerance nonlinear solver, default is 1e-15
    :param r_tol: relative tolerance nonlinear solver, default is 1e-8
    :param y_0: initial condition
    :param adaptive: boolean
    :param right_hand_side: function of the right-hand-side, i.e. dy/dt = rhs(y, t)

    Returns
    -------
    u: (Nx, Nt) ndarray
        Solution to the ODE at time t_vec; that is, y[:,j] is the
        computed solution corresponding to time t[j].

    """
    # initialize the solution matrix
    y_sol = np.zeros((len(y_0), len(param.t_vec)), dtype="complex128")
    y_sol[:, 0] = y_0

    # for-loop each time-step
    for tt in range(1, len(param.t_vec)):
        # print out the current time stamp
        print("\n time = ", param.t_vec[tt])
        if adaptive:
            # updated u (electron 1) parameter
            u_e_curr = updated_u(u_prev=param.u_e[-1],
                                 alpha_prev=param.alpha_e[-1],
                                 C00=y_sol[:, tt - 1][0],
                                 C10=y_sol[:, tt - 1][param.Nx_total])

            # updated alpha (electron 1) parameter
            alpha_e_curr = updated_alpha(alpha_prev=param.alpha_e[-1],
                                         C20=y_sol[:, tt - 1][2 * param.Nx_total],
                                         C10=y_sol[:, tt - 1][param.Nx_total],
                                         C00=y_sol[:, tt - 1][0])

            # electron 1 check mark
            if check_if_update_needed(u_s_curr=u_e_curr, u_s=param.u_e[-1], u_s_tol=param.u_tol,
                                      alpha_s_curr=alpha_e_curr, alpha_s=param.alpha_e[-1],
                                      alpha_s_tol=param.alpha_tol):
                print("updating u or alpha (electrons)")
                print("u_e = ", u_e_curr)
                print("alpha_e= ", alpha_e_curr)
                # get Hermite projection matrix
                P, case = get_projection_matrix(u_s_curr=u_e_curr, u_s=param.u_e[-1], alpha_s_curr=alpha_e_curr,
                                                alpha_s=param.alpha_e[-1], Nx_total=param.Nx_total, Nv=param.Nv,
                                                alpha_s_tol=param.alpha_tol, u_s_tol=param.u_tol)
                if case == 1:
                    print("tolerance met for u and alpha")
                    # update parameters
                    param.replace_alpha_e(alpha_e_curr=alpha_e_curr)
                    param.replace_u_e(u_e_curr=u_e_curr)

                elif case == 2:
                    print("tolerance met for u")
                    param.replace_u_e(u_e1_curr=u_e_curr)

                elif case == 3:
                    print("tolerance met for alpha")
                    param.replace_alpha_e(alpha_e_curr=alpha_e_curr)

                # project the previous timestamp results
                y_sol[:, tt - 1][:param.Nv * param.Nx_total] = P @ y_sol[:, tt - 1][:param.Nv * param.Nx_total]

            # update parameters electron 1
            param.add_alpha_e(alpha_e1_curr=param.alpha_e[-1])
            param.add_u_e(u_e1_curr=param.u_e[-1])

        y_sol[:, tt] = scipy.optimize.newton_krylov(F=lambda y: implicit_nonlinear_equation(y_new=y,
                                                                                            y_old=y_sol[:, tt - 1],
                                                                                            right_hand_side=right_hand_side,
                                                                                            dt=param.dt),
                                                    xin=y_sol[:, tt - 1],
                                                    maxiter=max_iter,
                                                    method='gmres',
                                                    f_tol=a_tol,
                                                    f_rtol=r_tol,
                                                    verbose=True)
    return y_sol, param

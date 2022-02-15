# 2021.04.14 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# 2021.11.22 updated by Tae Hoon Oh, E-mail: rozenk30@gmail.com

# Optimal control solved by orthogonal collocation in finite elements

import casadi as ca
import numpy as np
# from scipy.integrate import solve_ivp

import UtilityFunctions


class QMPC(object):

    def __init__(self, state_dim, input_dim, parameter_dim, dynamic_model,
                 path_wise_cost, terminal_cost, q_function, time_step=1,
                 state_min=None, state_max=None,
                 input_min=None, input_max=None,
                 parameter_min=None, parameter_max=None,
                 polynomial_order=3, element_number=1, clip_min=-0.02, clip_max=1.02, soft_coefficient=3):

        # path_wise_cost(x, u) -> scalar
        # terminal_cost(x, u) -> scalar
        # dynamic_model(x, u, p) -> list

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.parameter_dim = parameter_dim
        self.polynomial_order = polynomial_order
        self.element_number = element_number

        self.path_wise_cost = path_wise_cost
        self.terminal_cost = terminal_cost
        self.q_function = q_function
        self.time_step = time_step

        if state_min is None:
            state_min = -10**10*np.ones(self.state_dim)
        if state_max is None:
            state_max = 10**10*np.ones(self.state_dim)
        if input_min is None:
            input_min = -10**10*np.ones(self.input_dim)
        if input_max is None:
            input_max = 10**10*np.ones(self.input_dim)
        if parameter_min is None:
            parameter_min = -10**20*np.ones(self.parameter_dim)
        if parameter_max is None:
            parameter_max = 10**20*np.ones(self.parameter_dim)

        self.state_min = state_min
        self.state_max = state_max
        self.input_min = input_min
        self.input_max = input_max
        self.parameter_min = parameter_min
        self.parameter_max = parameter_max
        self.x_grad_scale = 1./(self.state_max - self.state_min)

        self.dynamic_model = dynamic_model

        # For numerical issues
        self.clip_min = clip_min  # clip bound, scaled value is supposed to be in [0, 1]
        self.clip_max = clip_max
        # relaxing the constraint of continuity equation in orthogonal collocation method
        self.soft_coefficient = soft_coefficient

    def _local_model_casadi(self):
        # Scaled in scaled out
        state_sym = ca.SX.sym('x', self.state_dim)
        input_sym = ca.SX.sym('u', self.input_dim)
        parameter_sym = ca.SX.sym('p', self.parameter_dim)

        state_sym_d = UtilityFunctions.descale(state_sym, self.state_min, self.state_max)
        input_sym_d = UtilityFunctions.descale(input_sym, self.input_min, self.input_max)
        parameter_sym_d = UtilityFunctions.descale(parameter_sym, self.parameter_min, self.parameter_max)
        xdot_ca, _, _, _ = self.dynamic_model(state_sym_d, input_sym_d, parameter_sym_d)
        # xdot_ca = []
        # for k in xdot:
        #     xdot_ca = ca.vertcat(xdot_ca, k)
        xdot_ca = np.multiply(xdot_ca, self.x_grad_scale)

        return xdot_ca, state_sym, input_sym, parameter_sym

    def _integration_casadi(self, initial_state, input_value, parameter_value, time_interval):
        # Scaled in, Scaled out (np.array 1dim in 1dim out)
        # Integrate ODE
        xdot_ca, state_sym, input_sym, parameter_sym = self._local_model_casadi()
        input_parameter = ca.SX.zeros(self.input_dim + self.parameter_dim)
        input_parameter[:self.input_dim] = input_sym
        input_parameter[self.input_dim:] = parameter_sym
        setting = {'x': state_sym, 'p': input_parameter, 'ode': xdot_ca}
        options = {'t0': 0, 'tf': time_interval}
        integral = ca.integrator('integrator', 'cvodes', setting, options)
        solution = integral(x0=initial_state, p=np.hstack((input_value, parameter_value)))
        x_plus = solution['xf']
        return np.squeeze(np.asarray(x_plus))

    def _ini_state_gen_casadi(self, initial_state, input_trajectory, parameter, horizon):
        ppo = self.polynomial_order + 1  # for short expression
        time_interval = (self.time_step / self.element_number) / ppo
        initial_state_trajectory = np.zeros((self.state_dim, horizon*self.element_number*ppo+1))
        initial_state_trajectory[:, 0] = initial_state
        for k in range(horizon):
            for kk in range(self.element_number):
                for kkk in range(ppo):
                    initial_state_trajectory[:, ppo*self.element_number*k + ppo*kk + kkk + 1] =\
                        self._integration_casadi(initial_state_trajectory[:, ppo*self.element_number*k + ppo*kk + kkk],
                                                 input_trajectory[:, k], parameter, time_interval)
        return initial_state_trajectory

    def control(self, initial_state, initial_input, parameter, horizon, terminal_flag, input_bound):

        # Descaled in descaled out
        # if terminal_flag: Use terminal cost
        # else: terminal cost : Q(S,A)

        s = UtilityFunctions.scale(initial_state, self.state_min, self.state_max)  # vector
        u = np.transpose(UtilityFunctions.scale(np.transpose(initial_input), self.input_min, self.input_max))  # matrix
        p = UtilityFunctions.scale(parameter, self.parameter_min, self.parameter_max)  # vector

        initial_state_trajectory = self._ini_state_gen_casadi(s, u, p, horizon)
        initial_state_trajectory = np.clip(initial_state_trajectory, self.clip_min, self.clip_max)  # clipping
        # print(initial_state_trajectory)

        # Symbolic set-up
        B, C, D = self._collocation_setup(self.polynomial_order, self.time_step/self.element_number)
        xdot_ca, state_sym, input_sym, parameter_sym = self._local_model_casadi()

        # Cost
        state_sym_d = UtilityFunctions.descale(state_sym, self.state_min, self.state_max)
        input_sym_d = UtilityFunctions.descale(input_sym, self.input_min, self.input_max)
        path_wise_cost_sym = self.path_wise_cost(state_sym_d, input_sym_d)

        if terminal_flag:
            terminal_cost_sym = self.terminal_cost(state_sym_d, input_sym_d)
        else:
            terminal_cost_sym = self.q_function(state_sym_d, input_sym_d)

        ode = ca.Function('Func', [state_sym, input_sym, parameter_sym], [xdot_ca], ['x', 'u', 'p'], ['xdot'])
        path_cost_func = ca.Function('Path_func', [state_sym, input_sym], [path_wise_cost_sym], ['x', 'u'], ['c'])
        ter_cost_func = ca.Function('Terminal_func', [state_sym, input_sym], [terminal_cost_sym], ['x', 'u'], ['tc'])
        # print(path_wise_cost_sym, terminal_cost_sym)

        # Start with an empty NLP
        w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []
        cost = 0

        # For plotting x and u given w
        x_plot, u_plot, p_plot = [], [], []

        # Parameter
        pk = ca.MX.sym('P', self.parameter_dim)
        w.append(pk)
        lbw = np.append(lbw, np.zeros((self.parameter_dim, 1)))
        ubw = np.append(ubw, np.ones((self.parameter_dim, 1)))
        w0 = np.append(w0, p)
        p_plot.append(pk)

        g.append(pk - p)
        lbg = np.append(lbg, np.zeros((self.parameter_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.parameter_dim, 1)))

        # for readability
        ppo = self.polynomial_order + 1
        for k in range(horizon):
            # Input
            uk = ca.MX.sym('U_' + str(k), self.input_dim)
            w.append(uk)
            lbw = np.append(lbw, np.zeros((self.input_dim, 1)))
            ubw = np.append(ubw, np.ones((self.input_dim, 1)))
            w0 = np.append(w0, u[:, k])
            u_plot.append(uk)

            # Constraint, to make g > 0
            g.append(uk - u[:, k] + input_bound*np.ones(self.input_dim))
            lbg = np.append(lbg, np.zeros((self.input_dim, 1)))
            ubg = np.append(ubg, 2*input_bound*np.ones((self.input_dim, 1)))

            for kk in range(self.element_number):
                # State, note that Xkj are not the state but the weight of the lagrange functions!
                xw = []
                xkj = ca.MX.sym('X_' + str(k) + '_' + str(kk) + '_' + str(0), self.state_dim)
                w.append(xkj)
                lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                w0 = np.append(w0, initial_state_trajectory[:, ppo*self.element_number*k + ppo*kk])
                xw.append(xkj)
                if kk == 0:
                    x_plot.append(xkj)

                for j in range(1, ppo):
                    xkj = ca.MX.sym('X_' + str(k) + '_' + str(kk) + '_' + str(j), self.state_dim)
                    xw.append(xkj)
                    w.append(xkj)
                    lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                    ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                    w0 = np.append(w0, initial_state_trajectory[:, ppo*self.element_number*k + ppo*kk + j])

                if k == 0 and kk == 0:  # Initial point
                    g.append(xw[0] - s)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))
                    cost += path_cost_func(xw[0], uk)

                elif kk == 0:  # Horizon change
                    g.append(xw[0] - x_at_terminal)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))
                    cost += path_cost_func(xw[0], uk)

                else:
                    # Continuity equation for state
                    g.append(xw[0] - x_at_terminal)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

                x_at_terminal = 0
                for j in range(ppo):
                    x_at_terminal += D[j]*xw[j]

                # Expression for the state derivative at the collocation point
                for j in range(1, ppo):
                    xd = C[0, j]*xw[0]
                    for r in range(self.polynomial_order):
                        xd += C[r+1, j]*xw[r+1]

                    # Append collocation equations
                    xdot = ode(xw[j], uk, pk)
                    g.append(xdot - xd)  # Dynamics Equation
                    # Relaxation
                    lbg = np.append(lbg, -10**(-self.soft_coefficient)*np.ones((self.state_dim, 1)))
                    ubg = np.append(ubg,  10**(-self.soft_coefficient)*np.ones((self.state_dim, 1)))

        xkend = ca.MX.sym('X_final_end', self.state_dim)
        w.append(xkend)
        lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
        ubw = np.append(ubw, np.ones((self.state_dim, 1)))
        w0 = np.append(w0, initial_state_trajectory[:, -1])
        x_plot.append(xkend)

        g.append(xkend - x_at_terminal)
        lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

        # Terminal_input
        ukend = ca.MX.sym('U_terminal', self.input_dim)
        w.append(ukend)
        lbw = np.append(lbw, np.zeros((self.input_dim, 1)))
        ubw = np.append(ubw, np.ones((self.input_dim, 1)))
        w0 = np.append(w0, u[:, -1])
        u_plot.append(ukend)

        # Terminal cost
        cost += ter_cost_func(xkend, ukend)
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)
        p_plot = ca.horzcat(*p_plot)

        # Create an NLP solver
        prob = {'f': cost, 'x': w, 'g': g}

        # "linear_solver": "ma27"
        # opts = {'print_time': True, 'ipopt': {"max_iter": 1000}, 'ipopt': {'print_level': 5}}
        # opts = {'ipopt': {'tol': 1e-6}, 'ipopt': {'acceptable_tol': 1e-4}, 'ipopt': {'print_level': 5}}
        # opts = {'ipopt': {'max_iter': 3000}}
        opts = {'print_time': False, "ipopt": {'print_level': 0}}  # For quite
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)
        # solver = ca.nlpsol('solver', 'sqpmethod', prob)  # For SQP w/o options

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot, p_plot], ['w'], ['x', 'u', 'p'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        constraint_value = sol['g']
        x_opt, u_opt, p_opt = trajectories(sol['x'])

        # to numpy array
        x_opt = x_opt.full()
        u_opt = u_opt.full()
        p_opt = p_opt.full()
        p_opt = p_opt.squeeze()
        cost = sol['f']

        descale_x_opt = np.transpose(UtilityFunctions.descale(np.transpose(x_opt), self.state_min, self.state_max))
        descale_u_opt = np.transpose(UtilityFunctions.descale(np.transpose(u_opt), self.input_min, self.input_max))
        descale_p_opt = UtilityFunctions.descale(p_opt, self.parameter_min, self.parameter_max)  # vector

        return descale_x_opt, descale_u_opt, descale_p_opt, cost, constraint_value

    @staticmethod
    def _collocation_setup(order, time_interval):
        # Degree of interpolating polynomial
        polynomial_order = order

        # Get collocation points
        evaluation_points = time_interval * np.append(0, ca.collocation_points(polynomial_order, 'legendre'))

        # Coefficients of the collocation equation
        C = np.zeros((polynomial_order + 1, polynomial_order + 1))  # For diff
        D = np.zeros(polynomial_order + 1)  # For value at 1
        B = np.zeros(polynomial_order + 1)  # For integration

        # Construct polynomial basis
        for j in range(polynomial_order + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            lagrange_polynomial = np.poly1d([1])
            for r in range(polynomial_order + 1):
                if r != j:
                    lagrange_polynomial *= np.poly1d([1, -evaluation_points[r]]) / (
                            evaluation_points[j] - evaluation_points[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = lagrange_polynomial(time_interval)

            # Evaluate the time derivative of the polynomial at all collocation points
            # to get the coefficients of the continuity equation
            derivate_polynomial = np.polyder(lagrange_polynomial)
            for r in range(polynomial_order + 1):
                C[j, r] = derivate_polynomial(evaluation_points[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            integral_polynomial = np.polyint(lagrange_polynomial)
            B[j] = integral_polynomial(time_interval)
        return B, C, D

'''
if __name__ == '__main__':
    # Test
    
    def path_cost(x, u):
        return 0

    def terminal_cost(x, u):
        return -x[2]

    def q_cost(x, u):
        return -x[2]

    def dynamics(xd, ud, pd):

        # deScaled in, descaled out
        time, x1, x2 = ca.vertsplit(xd)
        u1, u2 = ca.vertsplit(ud)
        p1 = pd  # later change

        dtdt = 1.
        dx1dt = -(u1 + 0.5*u1**2)*x1 + 0.5*u2*x2/(x1 + x2)
        dx2dt = u1*x1 - p1*u2*x1

        diff_values = ca.vertcat(dtdt, dx1dt, dx2dt)

        return diff_values, xd, ud, pd


    control = QMPC(3, 2, 1, dynamics, path_cost, terminal_cost, q_cost, time_step=0.1,
                                state_min=np.array([0., 0., 0.]), state_max=np.array([1., 1., 1.]),
                                input_min=np.array([0., 0.]), input_max=np.array([10., 10.]),
                                parameter_min=np.array([0.]), parameter_max=np.array([1.]),
                                )

    n = 10
    s = np.array([0., 1., 0.])
    u0 = np.zeros((2, 10))
    #u0 = np.array([[1.3, 1.8, 2.0, 2.5, 4.2, 5.0, 5.0, 5.0, 5.0, 5.0],
    #              [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]])
    p = np.array([0.7])
    # z = control._ini_state_gen_casadi(s, u, p, 10)
    # print(z)

    xt, ut, pt, et, ct = control.control(s, u0, p, n, True, 1.0)
    print(xt)
    print(ut)
    print(pt)
    print(et)
    print(ct)

    import matplotlib.pyplot as plt

    plt.plot(xt[0, :], xt[1, :], xt[0, :], xt[2, :])
    plt.show()

    plt.plot(xt[0, :], ut[0, :], xt[0, :], ut[1, :])
    plt.show()
'''


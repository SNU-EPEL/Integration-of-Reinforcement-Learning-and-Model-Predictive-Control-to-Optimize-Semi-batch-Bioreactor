# 2021.04.14 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Optimal control solved by orthogonal collocation in finite elements

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

import UtilityFunctions


class OptimalController(object):
    def __init__(self, polynomial_order, element_number, nn_weight):

        self.state_dim = 5
        self.input_dim = 1
        self.parameter_dim = 9
        self.polynomial_order = polynomial_order
        self.element_number = element_number

        self.terminal_cost_coeff = np.array([0.5, 0.5])
        self.input_cost_coeff = np.array([0.0005, 0.001])

        self.xmin = np.array([0, 0, 0.0001, 0, 0])
        self.xmax = np.array([230, 150, 25, 100, 110000])
        self.umin = np.array([10])
        self.umax = np.array([240])
        self.pmin = np.zeros(self.parameter_dim)  # Dummy
        self.pmax = (10**3)*np.ones(self.parameter_dim)  # Dummy
        self.x_grad_scale = 1./(self.xmax - self.xmin)
        self.nn_weight = nn_weight
        self.nn_layer = len(nn_weight)

        # Fixed parameters
        self.f_in_1 = 0.8436
        self.f_in_2 = 29.57
        self.sigma = 0.0015
        self.mu5 = 0.0027

    def local_model_casadi(self):
        # Put the ODE model here
        # Scaled in scaled out
        model_state = ca.SX.sym('x', self.state_dim)
        model_input = ca.SX.sym('u', self.input_dim)
        model_parameter = ca.SX.sym('p', self.parameter_dim)

        state_d = UtilityFunctions.descale(model_state, self.xmin, self.xmax)
        input_d = UtilityFunctions.descale(model_input, self.umin, self.umax)
        parameter_d = UtilityFunctions.descale(model_parameter, self.pmin, self.pmax)

        model_state_input = ca.vertcat(model_state, model_input)
        t, x, s, p, vol = ca.vertsplit(state_d)
        f_s = input_d  # ca.vertsplit(input_d) is not used since it is 1-dim
        mu1, mu2, mu3, mu4, k1, k2, s_star, f_evp, s_f = ca.vertsplit(parameter_d)

        dtdt = 1.
        dvdt = self.f_in_1*f_s + self.f_in_2 - f_evp*vol
        dxdt = mu1*s*x/(k1 + s) - x*dvdt/vol
        dsdt = -mu2*s*x/(k2 + s) - mu3*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) + f_s*s_f/vol - s*dvdt/vol
        dpdt = mu4*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) - self.mu5*p - p*dvdt/vol
        xdot = ca.vertcat(dtdt, dxdt, dsdt, dpdt, dvdt)
        xdot = np.multiply(xdot, self.x_grad_scale)

        return xdot, model_state_input, model_state, model_input, model_parameter

    def integration_casadi(self, initial_state, model_input, model_parameter, time_interval):
        # Scaled in, Scaled out (np.array 1dim in 1dim out)
        # Integrate ODE
        xdot, s, a, p = self.local_model_casadi(time_interval)
        input_parameter = ca.SX.zeros(self.input_dim + self.parameter_dim)
        input_parameter[:self.input_dim] = a
        input_parameter[self.input_dim:] = p
        setting = {'x': s, 'p': input_parameter, 'ode': xdot}
        options = {'t0': 0, 'tf': time_interval}
        I = ca.integrator('I', 'cvodes', setting, options)
        solution = I(x0=initial_state, p=np.hstack((model_input, model_parameter)))
        x_plus = solution['xf']
        return np.squeeze(np.asarray(x_plus))

    def initial_state_generator_casadi(
            self, initial_state, input_trajectory, parameter, step_time_interval, n, element_number):
        time_interval = (step_time_interval / element_number) / (self.polynomial_order + 1)
        initial_state_trajectory = np.zeros((self.state_dim, n*element_number*(self.polynomial_order+1)+1))
        initial_state_trajectory[:, 0] = initial_state
        for k in range(n):
            for kk in range(element_number):
                for kkk in range(self.polynomial_order+1):
                    initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k +
                                                (self.polynomial_order+1)*kk + kkk + 1] =\
                        self.integration(initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k +
                                                                     (self.polynomial_order+1)*kk + kkk],
                                         input_trajectory[:, k], parameter, time_interval)
        return initial_state_trajectory

    def action_value_casadi(self, x):
        out = x
        for k in range(self.nn_layer//2):
            out = activation(ca.transpose(ca.mtimes(ca.transpose(out), self.nn_weight[2*k])) + self.nn_weight[2*k+1])
        return out

    def local_model(self, t, model_state, model_input, model_parameter):

        state_d = UtilityFunctions.descale(model_state, self.xmin, self.xmax)
        input_d = UtilityFunctions.descale(model_input, self.umin, self.umax)
        parameter_d = UtilityFunctions.descale(model_parameter, self.pmin, self.pmax)

        t, x, s, p, vol = state_d
        f_s = input_d
        mu1, mu2, mu3, mu4, k1, k2, s_star, f_evp, s_f = parameter_d

        dtdt = 1.
        dvdt = self.f_in_1*f_s + self.f_in_2 - f_evp*vol
        dxdt = mu1*s*x/(k1 + s) - x*dvdt/vol
        dsdt = -mu2*s*x/(k2 + s) - mu3*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) + f_s*s_f/vol - s*dvdt/vol
        dpdt = mu4*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) - self.mu5*p - p*dvdt/vol
        xdot = np.array([dtdt, dxdt, dsdt, dpdt, dvdt])
        xdot = np.multiply(xdot, self.x_grad_scale)

        return xdot

    def integration(self, initial_state, model_input, model_parameter, time_interval):
        # Scaled in, Scaled out (np.array 1dim in 1dim out)
        # Integrate ODE by scipy
        tspan = [0, time_interval]
        sol = solve_ivp(fun=self.local_model, t_span=tspan, y0=initial_state, args=(model_input, model_parameter))
        sol_y = sol.y
        return sol_y[:, -1]

    def initial_state_generator(
            self, initial_state, input_trajectory, parameter, step_time_interval, n, element_number):
        time_interval = (step_time_interval / element_number) / (self.polynomial_order + 1)
        initial_state_trajectory = np.zeros((self.state_dim, n*element_number*(self.polynomial_order+1)+1))
        initial_state_trajectory[:, 0] = initial_state
        for k in range(n):
            for kk in range(element_number):
                for kkk in range(self.polynomial_order+1):
                    initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k
                                                + (self.polynomial_order+1)*kk + kkk + 1] = \
                        self.integration(initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k
                        + (self.polynomial_order+1)*kk + kkk], input_trajectory[:, k], parameter, time_interval)
        return initial_state_trajectory

    def control(self, initial_state, initial_input, parameter, step_time_interval, n, bound, softing, terminal):

        # Descaled in descaled out
        # if Terminal: Use terminal cost
        # else: terminal cost : Q(S,A)

        # zero_d, first_d, second_d <--- approximation... may need to generalize it
        # Think about the penalty on DNN

        s = UtilityFunctions.scale(initial_state, self.xmin, self.xmax)  # vector
        u = np.transpose(UtilityFunctions.scale(np.transpose(initial_input), self.umin, self.umax))  # matrix
        p = UtilityFunctions.scale(parameter, self.pmin, self.pmax)  # vector

        initial_state_trajectory = self.initial_state_generator(s, u, p, step_time_interval, n, self.element_number)
        initial_state_trajectory = np.clip(initial_state_trajectory, -0.2, 1.2)

        time_interval = step_time_interval/self.element_number

        # Symbolic set-up
        B, C, D = collocation_setup(self.polynomial_order, time_interval)
        xdot, model_state_input, model_state, model_input, model_para = self.local_model_casadi()

        # Cost
        stage_cost = self.input_cost_coeff[0]*model_input + self.input_cost_coeff[1]*model_input*model_input
        if terminal:
            terminal_cost = self.terminal_cost_coeff[0]*(1 - model_state[3]*model_state[4]) \
                            + self.terminal_cost_coeff[1]*(1 - model_state[3])
        else:
            terminal_cost = self.action_value_casadi(model_state_input)

        ode = ca.Function('Func', [model_state, model_input, model_para], [xdot], ['x', 'u', 'p'], ['xdot'])
        stage_cost_function = ca.Function('Cost_func', [model_state, model_input], [stage_cost], ['x', 'u'], ['c'])
        terminal_cost_function \
            = ca.Function('Terminal_Cost_func', [model_state, model_input], [terminal_cost], ['x', 'u'], ['tc'])

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        cost = 0
        g = []
        lbg = []
        ubg = []

        # For plotting x and u given w
        x_plot = []
        u_plot = []
        p_plot = []

        # Parameter
        pk = ca.MX.sym('P', self.parameter_dim)
        w.append(pk)
        lbw = np.append(lbw, np.zeros((self.parameter_dim, 1)))
        ubw = np.append(ubw, np.ones((self.parameter_dim,  1)))
        w0 = np.append(w0, p)
        p_plot.append(pk)

        g.append(pk - p)
        lbg = np.append(lbg, np.zeros((self.parameter_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.parameter_dim, 1)))

        for k in range(n):
            # Input
            uk = ca.MX.sym('U_' + str(k), self.input_dim)
            w.append(uk)
            lbw = np.append(lbw, np.zeros((self.input_dim, 1)))
            ubw = np.append(ubw, np.ones((self.input_dim, 1)))
            w0 = np.append(w0, u[:, k])

            # Constraint, to make g > 0
            u_plot.append(uk)
            g.append(uk - u[:, k] + bound*np.ones(self.input_dim))
            lbg = np.append(lbg, np.array([[0]]))
            ubg = np.append(ubg, np.array([[2*bound]]))

            for kk in range(self.element_number):
                # State, note that Xkj are not the state but the weight of the lagrange functions!
                xw = []
                xkj = ca.MX.sym('X_' + str(k) + '_' + str(kk) + '_' + str(0), self.state_dim)
                w.append(xkj)
                lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                w0 = np.append(w0, initial_state_trajectory[:, (self.polynomial_order+1)*self.element_number*k
                                                               + (self.polynomial_order+1)*kk])
                xw.append(xkj)
                if kk == 0:
                    x_plot.append(xkj)

                for j in range(1, self.polynomial_order+1):
                    xkj = ca.MX.sym('X_' + str(k) + '_' + str(kk) + '_' + str(j), self.state_dim)
                    xw.append(xkj)
                    w.append(xkj)
                    lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                    ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                    w0 = np.append(w0, initial_state_trajectory[:, (self.polynomial_order+1)*self.element_number*k
                                                                   + (self.polynomial_order+1)*kk + j])

                if k == 0 and kk == 0:  # Initial point
                    g.append(xw[0] - s)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))
                    cost += stage_cost_function(xw[0], uk)

                elif kk == 0:  # Horizon change
                    g.append(xw[0] - x_at_terminal)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))
                    cost += stage_cost_function(xw[0], uk)

                else:
                    # Continuity equation for state
                    g.append(xw[0] - x_at_terminal)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

                x_at_terminal = 0
                for j in range(self.polynomial_order + 1):
                    x_at_terminal += D[j]*xw[j]

                # Expression for the state derivative at the collocation point
                for j in range(1, self.polynomial_order + 1):
                    xd = C[0, j]*xw[0]
                    for r in range(self.polynomial_order):
                        xd += C[r+1, j]*xw[r+1]

                    # Append collocation equations
                    xdot = ode(xw[j], uk, pk)
                    g.append(xdot - xd)  # Dynamics Equation
                    # Relaxation
                    lbg = np.append(lbg, -10**(-softing)*np.ones((self.state_dim, 1)))
                    ubg = np.append(ubg,  10**(-softing)*np.ones((self.state_dim, 1)))

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
        cost += terminal_cost_function(xkend, ukend)

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
        opts = {'print_time': False, "ipopt": {'print_level': 0}}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)
        # solver = ca.nlpsol('solver', 'sqpmethod', prob)

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
        error = sol['f']

        descale_x_opt = np.transpose(UtilityFunctions.descale(np.transpose(x_opt), self.xmin, self.xmax))  # matrix
        descale_u_opt = np.transpose(UtilityFunctions.descale(np.transpose(u_opt), self.umin, self.umax))  # matrix
        descale_p_opt = UtilityFunctions.descale(p_opt, self.pmin, self.pmax)  # vector

        return descale_x_opt, descale_u_opt, descale_p_opt, error, constraint_value


def collocation_setup(order, time_interval):
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


def activation(x):
    x_len, _ = x.shape
    for k in range(x_len):
        x[k, :] = ca.log(1 + x[k, :]*x[k, :])
        y = x
    return y


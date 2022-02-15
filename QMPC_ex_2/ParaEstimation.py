# 2021.04.14 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Plant simulation with reference trajectory

import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp

import UtilityFunctions

# Use the lumped model
# state : t, X, S, P, V
# input : F_s ( F_s + F_oil*5/3 )
# parameter : mu1, mu2, mu3, mu4, k1, k2, s_star, F_evp


class ParameterEstimator(object):
    def __init__(self, order, element_number, pmin, pmax):

        self.state_dim = 5
        self.input_dim = 1
        self.parameter_dim = 9
        self.polynomial_order = order
        self.element_number = element_number

        self.weight = np.diag([0, 10, 10, 10, 1*10**-4])

        self.xmin = np.array([0, 0, 0.0001, 0, 0])
        self.xmax = np.array([230, 300, 20, 120, 150000])
        self.umin = np.array([30])
        self.umax = np.array([300])
        self.total_pmin = pmin  # trajecotry, matrix
        self.total_pmax = pmax  # trajectory, matrix
        self.x_grad_scale = 1./(self.xmax - self.xmin)

        # Fixed parameters
        self.f_in_1 = 0.8436
        self.f_in_2 = 29.57
        self.sigma = 0.0015
        self.mu5 = 0.0027

    def local_model_casadi(self):

        model_state = ca.SX.sym('x', self.state_dim)
        model_input = ca.SX.sym('u', self.input_dim)
        model_parameter = ca.SX.sym('p', self.parameter_dim)

        state_d = UtilityFunctions.descale(model_state, self.xmin, self.xmax)
        input_d = UtilityFunctions.descale(model_input, self.umin, self.umax)
        parameter_d = UtilityFunctions.descale(model_parameter, self.pmin, self.pmax)

        t, x, s, p, vol = ca.vertsplit(state_d)
        f_s = input_d   # ca.vertsplit(input_d) is not used since it is 1-dim
        mu1, mu2, mu3, mu4, k1, k2, s_star, f_evp, s_f = ca.vertsplit(parameter_d)

        dtdt = 1.
        dvdt = self.f_in_1*f_s + self.f_in_2 - f_evp*vol
        dxdt = mu1*s*x/(k1 + s) - x*dvdt/vol
        dsdt = -mu2*s*x/(k2 + s) - mu3*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) + f_s*s_f/vol - s*dvdt/vol
        dpdt = mu4*x*ca.exp(-1/2*((s - s_star)/self.sigma)**2) - self.mu5*p - p*dvdt/vol
        xdot = ca.vertcat(dtdt, dxdt, dsdt, dpdt, dvdt)
        xdot = np.multiply(xdot, self.x_grad_scale)

        return xdot, model_state, model_input, model_parameter

    def integration_casadi(self, initial_state, model_input, model_parameter, time_interval):
        # Scaled in, Scaled out (np.array 1dim in 1dim out)
        # Integrate ODE
        xdot, s, a, p = self.local_model_casadi()
        input_parameter = ca.SX.zeros(self.input_dim + self.parameter_dim)
        input_parameter[:self.input_dim] = a
        input_parameter[self.input_dim:] = p
        setting = {'x': s, 'p': input_parameter, 'ode': xdot}
        options = {'t0': 0, 'tf': time_interval}
        I = ca.integrator('I', 'cvodes', setting, options)
        solution = I(x0 = initial_state, p = np.hstack((model_input, model_parameter)))
        x_plus = solution['xf']
        return np.squeeze(np.asarray(x_plus))

    def initial_state_generator_casadi(
            self, initial_state, input_trajectory, model_parameter, step_time_interval, n, element_number):
        time_interval = (step_time_interval / element_number) / (self.polynomial_order + 1)
        initial_state_trajectory = np.zeros((self.state_dim, n*element_number*(self.polynomial_order+1)+1))
        initial_state_trajectory[:, 0] = initial_state
        for k in range(n):
            for kk in range(element_number):
                for kkk in range(self.polynomial_order+1):
                    initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k +
                                                (self.polynomial_order+1)*kk + kkk + 1] = \
                        self.integration_casadi(initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k +
                                                                            (self.polynomial_order+1)*kk + kkk],
                                                input_trajectory[:,k], model_parameter, time_interval)
        return initial_state_trajectory

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
            self, initial_state, input_trajectory, model_parameter, step_time_interval, n, element_number):
        time_interval = (step_time_interval / element_number) / (self.polynomial_order + 1)
        initial_state_trajectory = np.zeros((self.state_dim, n*element_number*(self.polynomial_order+1)+1))
        initial_state_trajectory[:, 0] = initial_state
        for k in range(n):
            for kk in range(element_number):
                for kkk in range(self.polynomial_order+1):
                    initial_state_trajectory[:, (self.polynomial_order+1)*element_number*k
                                                + (self.polynomial_order+1)*kk + kkk + 1] = \
                        self.integration(initial_state_trajectory[:,(self.polynomial_order+1)*element_number*k
                        + (self.polynomial_order+1)*kk + kkk], input_trajectory[:, k], model_parameter, time_interval)
        return initial_state_trajectory

    def estimation(self, state_data, input_data, initial_parameter, step_time_interval, time_index, softing):

        # Descaled in, Descaled out
        # Instead of arriving cost, the initial state is imposed as constraint

        # parameter bound change
        self.pmin = self.total_pmin[:, time_index]
        self.pmax = self.total_pmax[:, time_index]

        n = input_data.shape[1]
        so = np.transpose(UtilityFunctions.scale(np.transpose(state_data), self.xmin, self.xmax))  # matrix
        a = np.transpose(UtilityFunctions.scale(np.transpose(input_data), self.umin, self.umax))  # matrix
        p = UtilityFunctions.scale(initial_parameter, self.pmin, self.pmax)  # vector
        s = so[:, 0:n]
        o = so[:, 1:n+1]

        time_interval = step_time_interval/self.element_number
        B, C, D = collocation_setup(self.polynomial_order, time_interval)
        xdot, model_state, model_input, model_parameter = self.local_model_casadi()
        output = ca.SX.sym('y', self.state_dim)

        # Initial_state is generated by forward simulation
        initial_state = self.initial_state_generator(s[:, 0], a, p, step_time_interval, n, self.element_number)

        stage_cost = ca.mtimes(ca.transpose(ca.mtimes(self.weight, model_state - output)), model_state - output)

        ode = ca.Function('Func', [model_state, model_input, model_parameter], [xdot], ['x', 'u', 'p'], ['xdot'])
        cost_function = ca.Function('Cost_func', [model_state, output], [stage_cost], ['x', 'y'], ['c'])

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
        y_plot = []

        # Parameter
        pk = ca.MX.sym('pk', self.parameter_dim)
        w.append(pk)
        lbw = np.append(lbw, np.zeros((self.parameter_dim, 1)))
        ubw = np.append(ubw, np.ones((self.parameter_dim, 1)))
        w0 = np.append(w0, p)
        p_plot.append(pk)

        for k in range(n):
            # Input
            uk = ca.MX.sym('u_' + str(k), self.input_dim)
            w.append(uk)
            lbw = np.append(lbw, np.zeros((self.input_dim, 1)))
            ubw = np.append(ubw, np.ones((self.input_dim, 1)))
            w0 = np.append(w0, a[:, k])
            u_plot.append(uk)

            # Input constraint
            g.append(uk - a[:, k])
            lbg = np.append(lbg, np.zeros((self.input_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.input_dim, 1)))

            # Output
            yk = ca.MX.sym('y_' + str(k), self.state_dim)
            w.append(yk)
            lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
            ubw = np.append(ubw, np.ones((self.state_dim, 1)))
            w0 = np.append(w0, o[:, k])
            y_plot.append(yk)

            # Output constraint
            g.append(yk - o[:, k])
            lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
            ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

            for kk in range(self.element_number):
                # State, note that Xkj are not the state but the weight of the lagrange functions!
                xw = []
                xkj = ca.MX.sym('x_' + str(k) + '_' + str(kk) + '_' + str(0), self.state_dim)
                w.append(xkj)
                lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                w0 = np.append(w0, initial_state[:, (self.polynomial_order+1)*self.element_number*k + (self.polynomial_order + 1)*kk])
                xw.append(xkj)
                if kk == 0:
                    x_plot.append(xkj)

                for j in range(1, self.polynomial_order+1):
                    xkj = ca.MX.sym('X_' + str(k) + '_' + str(kk) + '_' + str(j), self.state_dim)
                    xw.append(xkj)
                    w.append(xkj)
                    lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
                    ubw = np.append(ubw, np.ones((self.state_dim, 1)))
                    w0 = np.append(w0, initial_state[:, (self.polynomial_order+1)*self.element_number*k + (self.polynomial_order+1)*kk + j])

                if k == 0 and kk == 0:
                    g.append(xw[0] - s[:, 0])
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))
                else:
                    g.append(xw[0] - x_at_terminal)
                    lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
                    ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

                x_at_terminal = 0

                for j in range(self.polynomial_order + 1):
                    x_at_terminal += D[j] * xw[j]

                if kk == self.element_number - 1:  # Horizon change
                    cost += cost_function(x_at_terminal, yk)

                # Expression for the state derivative at the collocation point
                for j in range(1, self.polynomial_order + 1):
                    xd = C[0, j]*xw[0]
                    for r in range(self.polynomial_order):
                        xd += C[r+1, j]*xw[r+1]

                    # Append collocation equations
                    xdot = ode(xw[j], uk, pk)
                    g.append(xdot - xd)  # Dynamics Equality
                    # Relaxation
                    lbg = np.append(lbg, -10**(-softing)*np.ones((self.state_dim, 1)))
                    ubg = np.append(ubg,  10**(-softing)*np.ones((self.state_dim, 1)))

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)
        p_plot = ca.horzcat(*p_plot)
        y_plot = ca.horzcat(*y_plot)

        # Create an NLP solver
        prob = {'f': cost, 'x': w, 'g': g}

        # "linear_solver": "ma27"
        # 'ipopt': {'alpha_for_y': 'min'}
        # 'ipopt': {'acceptable_tol': 1e-4}
        # 'ipopt': {'expect_infeasible_problem_ctol': 0.01}
        # 'ipopt': {'tol': 1e-6}
        # 'ipopt': {'expect_infeasible_problem': 'yes'}
        # 'ipopt': {'print_level': 5}
        opts = {'ipopt': {'max_iter': 3000}}
        # solver = ca.nlpsol('solver', 'sqpmethod', prob)
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot, p_plot, y_plot], ['w'], ['x', 'u', 'p', 'y'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, u_opt, p_opt, y_opt = trajectories(sol['x'])
        # to numpy array
        x_opt = x_opt.full()
        u_opt = u_opt.full()
        p_opt = p_opt.full()
        p_opt = p_opt.squeeze()
        y_opt = y_opt.full()
        error = sol['f']

        descale_x_opt = np.transpose(UtilityFunctions.descale(np.transpose(x_opt), self.xmin, self.xmax))  # matrix
        descale_u_opt = np.transpose(UtilityFunctions.descale(np.transpose(u_opt), self.umin, self.umax))  # matrix
        descale_p_opt = UtilityFunctions.descale(p_opt, self.pmin, self.pmax)  # vector
        descale_y_opt = np.transpose(UtilityFunctions.descale(np.transpose(y_opt), self.xmin, self.xmax))  # matrix

        return descale_x_opt, descale_u_opt, descale_p_opt, descale_y_opt, error


def collocation_setup(order, time_interval):

    # Degree of interpolating polynomial
    polynomial_order = order

    # Get collocation points
    evaluation_points = time_interval*np.append(0, ca.collocation_points(polynomial_order, 'legendre'))

    # Coefficients of the collocation equation
    C = np.zeros((polynomial_order+1, polynomial_order+1))  # For diff
    D = np.zeros(polynomial_order+1)  # For value at 1
    B = np.zeros(polynomial_order+1)  # For integration

    # Construct polynomial basis
    for j in range(polynomial_order+1):
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        lagrange_polynomial = np.poly1d([1])
        for r in range(polynomial_order+1):
            if r != j:
                lagrange_polynomial *= np.poly1d([1, -evaluation_points[r]]) / (evaluation_points[j] - evaluation_points[r])

        # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
        D[j] = lagrange_polynomial(time_interval)

        # Evaluate the time derivative of the poly at all collocation points
        # to get the coefficients of the continuity equation
        derivate_polynomial = np.polyder(lagrange_polynomial)
        for r in range(polynomial_order+1):
            C[j, r] = derivate_polynomial(evaluation_points[r])

        # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
        integral_polynomial = np.polyint(lagrange_polynomial)
        B[j] = integral_polynomial(time_interval)
    return B, C, D

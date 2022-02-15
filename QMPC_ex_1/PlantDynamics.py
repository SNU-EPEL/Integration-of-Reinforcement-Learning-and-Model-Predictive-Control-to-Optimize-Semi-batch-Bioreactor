# 2021.11.19 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Plant dynamics, reference
# "Reinforcement learning for batch bioprocess optimization (2020)"

import numpy as np
import casadi as ca

import UtilityFunctions


# state (3): Time, x1, x2
# input (2): u1, u2
# para  (1): p1


class PlantDynamicsSimulator:
    def __init__(self, seed, state_perturb, parameter_perturb):
        # Seed
        self.seed = seed
        np.random.seed(self.seed)

        self.state_dimension = 3
        self.input_dimension = 2
        self.para_dimension = 1

        self.initial_state = np.array([0., 1., 0.])
        # Parameter perturbation
        # self.initial_perturb_coefficient = np.array([0., 1., 1.])

        self.initial_input = np.array([0., 0.])
        self.initial_time = 0.
        self.terminal_time = 1.  # Scaled time
        self.time_interval = 0.05
        self.horizon_length = int((self.terminal_time - self.initial_time) / self.time_interval)  # episode length

        self.xmin = np.array([0., 0., 0.])
        self.xmax = np.array([1., 1., 1.])
        self.umin = np.array([0., 0.])
        self.umax = np.array([6., 6.])
        self.pmin = np.array([0.])
        self.pmax = np.array([1.])
        #  y = ax + b, a = 1./(xmax - xmin) , dydt = a*dxdt = af(x) = af(x)
        self.scale_grad = 1. / (self.xmax - self.xmin)

        # State perturbation
        if state_perturb:
            self.state_perturb_std = 0.02
        else:
            self.state_perturb_std = 0.

        # Parameter perturbation
        if parameter_perturb:
            self.parameter_perturb_std = 0.3
        else:
            self.parameter_perturb_std = 0.

    def reset(self):
        plant_state, plant_input = self.initial_state, self.initial_input
        return plant_state, plant_input

    def step(self, plant_state, plant_input, plant_para):
        # Descaled in, Descaled out
        # Integrate ODE

        scaled_state = UtilityFunctions.scale(plant_state, self.xmin, self.xmax)
        scaled_input = UtilityFunctions.scale(plant_input, self.umin, self.umax)
        scaled_para = UtilityFunctions.scale(plant_para, self.pmin, self.pmax)

        scaled_state = np.clip(scaled_state, 0, 1)
        scaled_input = np.clip(scaled_input, 0, 1)

        plant_state = ca.SX.sym('state', self.state_dimension)
        plant_input = ca.SX.sym('input', self.input_dimension)
        plant_para = ca.SX.sym('para', self.para_dimension)
        plant_input_para = ca.SX.zeros(self.input_dimension + self.para_dimension)
        plant_input_para[:self.input_dimension] = plant_input
        plant_input_para[self.input_dimension:] = plant_para

        xd = UtilityFunctions.descale(plant_state, self.xmin, self.xmax)
        ud = UtilityFunctions.descale(plant_input, self.umin, self.umax)
        pd = UtilityFunctions.descale(plant_para, self.pmin, self.pmax)

        xdot, x, u, p = self.dynamics(xd, ud, pd)
        xdot = np.multiply(xdot, self.scale_grad)  # Because of scaling
        setting = {'x': plant_state, 'p': plant_input_para, 'ode': xdot}
        options = {'t0': 0, 'tf': self.time_interval}  # "nonlinear_solver_iteration": "functional"
        integrate = ca.integrator('integrate', 'idas', setting, options)  # cvodes
        solution = integrate(x0=scaled_state, p=np.hstack((scaled_input, scaled_para)))
        x_plus = np.squeeze(np.array(solution['xf']))

        descaled_x_plus = UtilityFunctions.descale(x_plus, self.xmin, self.xmax)
        x_noise = self.state_perturb_std*np.random.rand(self.state_dimension)
        x_noise[0] = 0  # Zero noise for time, change later  ###
        descaled_x_plus = descaled_x_plus*(1 + x_noise)
        descaled_x_plus = np.clip(descaled_x_plus, self.xmin, self.xmax)
        return descaled_x_plus

    @staticmethod
    def dynamics(xd, ud, pd):

        # descaled in, descaled out
        time, x1, x2 = ca.vertsplit(xd)
        u1, u2 = ca.vertsplit(ud)
        p1 = pd  # later change

        dtdt = 1.
        dx_1dt = -(u1 + 0.5*u1**2)*x1 + 0.5*u2*x2/(x1 + x2 + 0.1)
        dx_2dt = u1*x1 - p1*u2*x1

        diff_values = ca.vertcat(dtdt, dx_1dt, dx_2dt)

        return diff_values, xd, ud, pd


class PlantSimulator(PlantDynamicsSimulator):
    # Noise 구현은 여기서 해야 될듯 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, seed, state_perturb, parameter_perturb):
        super().__init__(seed, state_perturb, parameter_perturb)

    def step(self, plant_state, plant_input, plant_para, terminal_flag):
        descaled_x_plus = super().step(plant_state, plant_input, plant_para)
        if terminal_flag:
            cost = self.terminal_cost(plant_state, plant_input)
        else:
            cost = self.path_wise_cost(plant_state, plant_input)

        return descaled_x_plus, plant_input, cost
    
    @ staticmethod
    def path_wise_cost(state, action):
        return 0
    
    @ staticmethod
    def terminal_cost(state, action):
        return 1-state[2]


if __name__ == "__main__":

    # Test
    import matplotlib.pyplot as plt

    plant = PlantDynamicsSimulator(10, False, False)
    plant2 = PlantSimulator(10, False, False)

    x = np.zeros((3, 11))
    u = np.array([[1.3, 1.8, 2.0, 2.5, 4.2, 5.0, 5.0, 5.0, 5.0, 5.0],
                  [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]])
    p = np.array([0.7])
    x0, u0 = plant2.reset()
    x[:, 0] = x0
    for k in range(10):
        if k < 9:
            terminal_flag = False
        else:
            terminal_flag = True
        x[:, k+1], u_now, cost = plant2.step(x[:, k], u[:, k], p,  terminal_flag)
        print(terminal_flag, cost)

    plt.plot(x[0, :], x[1, :], x[0, :], x[2, :])
    plt.show()

    plt.plot(x[0, 0:10], u[0, :], x[0, 0:10], u[1, :])
    plt.show()






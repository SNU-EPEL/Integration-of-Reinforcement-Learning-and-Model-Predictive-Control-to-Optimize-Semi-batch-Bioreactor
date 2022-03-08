# 2021.04.14 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Plant simulation for a single step
# Need to change by seeing the simulator #####################################################

import numpy as np

from PlantPID import PID
from PlantDynamics import PlantDynamicsEquation
import UtilityFunctions

# state(29): Time,A_0,A_1,A_3,A_4,Integral_X,S,P,V,T,H,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,nm,DO,DCO2,viscosity,PAA,O2,CO2,Ni
# input(11): F_S, F_oil, F_PAA, F_a, F_b, F_w, F_g, F_c, F_h, Pressure, NH3

UMIN = 10
UMAX = 240
INPUT_COST_COEFFICIENT = 0.0005
INPUT_COST_COEFFICIENT2 = 0.001
STATE_COST_COEFFICIENT = 0.5
STATE_COST_COEFFICIENT2 = 0.5


class Simulator(object):
    def __init__(self, seed, ini_perturb, para_perturb):

        # Define the plant with seed
        self.plant = PlantDynamicsEquation(seed=seed, ini_perturb=ini_perturb, para_perturb=para_perturb)

        # Define the target value for PID
        self.temp_ref = 298.
        self.pH_ref = 6.5
        self.paa_ref = 1000  # 200 - 2000
        self.viscosity_ref = 100
        self.ni_upper = 400  # > 150
        self.dissolved_o2_upper = 8  # > 6.6

        # Define the PID controller
        self.PID_cooling = PID(4000, 80, 1000, 2500)
        self.PID_heating = PID(5, 0, 0, 0)
        self.PID_acid = PID(0.01, 0, 0, 0)
        self.PID_base = PID(1.5, 0.1, 0, 0)
        self.PID_water = PID(1, 0, 0, 0)
        self.PID_paa = PID(0.12, 0, 0, 0)
        self.PID_dissolved_o2 = PID(0.08, 0, 0, 0)

        self.previous_input = np.zeros((self.plant.input_dimension, self.plant.horizon_length + 1))

    def reset(self):
        x0, u0 = self.plant.reset()
        return x0, u0

    def step(self, plant_state, plant_input, time_index):

        if time_index > 160:
            self.PID_cooling = PID(1200, 0, 2000, 0)  # Gain scheduling

        if time_index == 0:
            previous_input = np.zeros(self.plant.input_dimension)
        else:
            previous_input = self.previous_input[:, time_index-1]

        time, a_0, a_1, a_3, a_4, integral_x, s, p, v, temp, hydro_ion, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, \
        dissolved_o2, dissolved_co2, viscosity, paa, o2, co2, ni = plant_state
        f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3 = plant_input
        f_s_p, f_oil_p, f_paa_p, f_a_p, f_b_p, f_w_p, f_g_p, f_c_p, f_h_p, pressure_p, nh3_p = previous_input
        f_s_l, f_oil_l, f_paa_l, f_a_l, f_b_l, f_w_l, f_g_l, f_c_l, f_h_l, pressure_l, nh3_l = self.plant.umin
        f_s_u, f_oil_u, f_paa_u, f_a_u, f_b_u, f_w_u, f_g_u, f_c_u, f_h_u, pressure_u, nh3_u = self.plant.umax

        # PID
        if temp > self.temp_ref - 0.01:
            f_c = self.PID_cooling.control(temp - self.temp_ref, self.plant.time_interval)
            f_h = f_h_l
        elif temp < self.temp_ref - 0.1:
            f_c = f_c_l
            f_h = self.PID_heating.control(self.temp_ref - temp, self.plant.time_interval)
        else:
            f_c = f_c_p
            f_h = f_h_p

        if -np.log10(hydro_ion) > self.pH_ref + 0.03:
            f_a = self.PID_acid.control(-self.pH_ref - np.log10(hydro_ion), self.plant.time_interval)
            f_b = f_b_l
        elif -np.log10(hydro_ion) < self.pH_ref:
            f_a = f_a_l
            f_b = self.PID_base.control(np.log10(hydro_ion) + self.pH_ref, self.plant.time_interval)
        else:
            f_a = f_a_p
            f_b = f_b_p

        if paa < 400:
            f_paa = f_paa_u
        elif paa > 1000:
            f_paa = f_paa_l
        else:
            f_paa = self.PID_paa.control(self.paa_ref - paa, self.plant.time_interval)
            f_paa = np.clip(f_paa, f_paa_l, f_paa_u)

        if viscosity > self.viscosity_ref:
            f_w = self.PID_water.control(viscosity - self.viscosity_ref, self.plant.time_interval)
        else:
            f_w = f_w_l

        if ni < self.ni_upper:
            nh3 = nh3_u

        if dissolved_o2 < self.dissolved_o2_upper:
            f_g = f_g + self.PID_dissolved_o2.control(self.dissolved_o2_upper - dissolved_o2, self.plant.time_interval)

        input_now = np.array([f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3])
        self.previous_input[:, time_index] = input_now
        next_state = self.plant.step(plant_state, input_now)
        next_state = np.squeeze(next_state)

        if time_index == 460:  # Terminal
            scaled_state = UtilityFunctions.scale(next_state, self.plant.xmin, self.plant.xmax)
            reward = STATE_COST_COEFFICIENT*(1 - scaled_state[7]*scaled_state[8]) \
                     + STATE_COST_COEFFICIENT2*(1 - scaled_state[7])
        else:
            scaled_input = UtilityFunctions.scale(
                UtilityFunctions.plant_input_to_local_input(input_now), UMIN, UMAX)
            reward = INPUT_COST_COEFFICIENT*scaled_input + INPUT_COST_COEFFICIENT2*scaled_input*scaled_input

        return next_state, input_now, reward


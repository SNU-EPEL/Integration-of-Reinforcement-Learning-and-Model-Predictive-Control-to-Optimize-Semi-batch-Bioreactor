# 2021.04.14 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Plant simulation with reference trajectory

import os
import numpy as np

import UtilityFunctions
from PlantPID import PID
from PlantDynamics import PlantDynamicsEquation

# state(29): Time,A_0,A_1,A_3,A_4,Integral_X,S,P,V,T,H,n0,n1,n2,n3,n4,n5,n6,n7,n8,n9,nm,DO,DCO2,viscosity,PAA,O2,CO2,Ni
# input(11): F_S, F_oil, F_PAA, F_a, F_b, F_w, F_g, F_c, F_h, Pressure, NH3

total_iterative = 100

base_seed = 100
ini_perturb = False
para_perturb = False

# Directory
directory = os.getcwd()

# Define the target value for PID
temp_ref = 298.
pH_ref = 6.5
paa_ref = 1000
viscosity_ref = 100
ni_upper = 400
dissolved_o2_upper = 8

# Reward coefficient
input_cost_coeff = 0.00005 #0.00005
state_cost_coeff = 0.000001

for epi in range(total_iterative):

    # Define the plant
    plant = PlantDynamicsEquation(seed=base_seed + epi, ini_perturb=ini_perturb, para_perturb=para_perturb)

    horizon_length = plant.horizon_length

    # Define the PID controller
    PID_cooling = PID(4000, 80, 1000, 2500)
    PID_heating = PID(5, 0, 0, 0)
    PID_acid = PID(0.01, 0, 0, 0)
    PID_base = PID(1.5, 0.1, 0, 0)
    PID_water = PID(1, 0, 0, 0)
    PID_paa = PID(0.12, 0, 0, 0)
    PID_dissolved_o2 = PID(0.08, 0, 0, 0)

    # Initial state and input
    plant_state, _ = plant.reset()
    u = np.loadtxt('Reference_input.txt')
    # u[0:2, :] = np.multiply(u[0:2, :], 1 + 0.05*(np.random.uniform(-1, 1, plant.input_dimension)  # 5% perturbation

    '''
    # Other substrate input trajectory
    input_trajectory = np.array([np.loadtxt('DDP_current_control.txt')])
    local_input_for_plant = UtilityFunctions.local_input_to_plant_input(input_trajectory)
    u[0, :] = local_input_for_plant[0, :]
    u[1, :] = local_input_for_plant[1, :]
    '''

    # Data record - descaled values
    state_record = np.zeros((plant.state_dimension, horizon_length + 1))
    input_record = np.zeros((plant.input_dimension, horizon_length + 1))
    reward_record = np.zeros((1, horizon_length + 1))
    local_state_record = np.zeros((5, horizon_length + 1))
    local_input_record = np.zeros((1, horizon_length + 1))

    # Simulation
    state_record[:, 0] = plant_state
    local_state_record[:, 0] = UtilityFunctions.plant_state_to_local_state(plant_state)

    for time_step in range(horizon_length):
        print('Epi, time step: ', epi, time_step)

        if time_step > 160:
            PID_cooling = PID(1200, 0, 2000, 0)  # Gain scheduling
            PID_cooling.reset()

        if time_step > 1:
            previous_input = input_record[:, time_step - 1]
        else:
            previous_input = np.zeros(plant.input_dimension)

        time, a_0, a_1, a_3, a_4, integral_x, s, p, v, temp, hydro_ion, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, nm, \
        dissolved_o2, dissolved_co2, viscosity, paa, o2, co2, ni = plant_state
        f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3 = u[:, time_step]

        f_s_p, f_oil_p, f_paa_p, f_a_p, f_b_p, f_w_p, f_g_p, f_c_p, f_h_p, pressure_p, nh3_p = previous_input
        f_s_l, f_oil_l, f_paa_l, f_a_l, f_b_l, f_w_l, f_g_l, f_c_l, f_h_l, pressure_l, nh3_l = plant.umin
        f_s_u, f_oil_u, f_paa_u, f_a_u, f_b_u, f_w_u, f_g_u, f_c_u, f_h_u, pressure_u, nh3_u = plant.umax

        # PID
        if temp > temp_ref - 0.01:
            f_c = PID_cooling.control(temp - temp_ref, plant.time_interval)
            f_h = f_h_l
        elif temp < temp_ref - 0.1:
            f_c = f_c_l
            f_h = PID_heating.control(temp_ref - temp, plant.time_interval)
        else:
            f_c = f_c_p
            f_h = f_h_p

        if -np.log10(hydro_ion) > pH_ref + 0.03:
            f_a = PID_acid.control(-pH_ref - np.log10(hydro_ion), plant.time_interval)
            f_b = f_b_l
        elif -np.log10(hydro_ion) < pH_ref:
            f_a = f_a_l
            f_b = PID_base.control(np.log10(hydro_ion) + pH_ref, plant.time_interval)
        else:
            f_a = f_a_p
            f_b = f_b_p

        if paa < 400:
            f_paa = f_paa_u
        elif paa > 1000:
            f_paa = f_paa_l
        else:
            f_paa = PID_paa.control(paa_ref - paa, plant.time_interval)
            f_paa = np.clip(f_paa, f_paa_l, f_paa_u)

        if viscosity > viscosity_ref:
            f_w = PID_water.control(viscosity - viscosity_ref, plant.time_interval)
        else:
            f_w = f_w_l

        if ni < ni_upper:
            nh3 = nh3_u

        if dissolved_o2 < dissolved_o2_upper:
            f_g = f_g + PID_dissolved_o2.control(dissolved_o2_upper - dissolved_o2, plant.time_interval)

        u[:, time_step] = np.array([f_s, f_oil, f_paa, f_a, f_b, f_w, f_g, f_c, f_h, pressure, nh3])

        input_record[:, time_step] = u[:, time_step]
        local_input_record[:, time_step] = UtilityFunctions.plant_input_to_local_input(u[:, time_step])
        reward_record[:, time_step] = input_cost_coeff*(u[0, time_step] + 5/3*u[1, time_step])
        plant_state = plant.step(plant_state, u[:, time_step])
        plant_state = np.squeeze(plant_state)

        state_record[:, time_step + 1] = plant_state
        local_state_record[:, time_step + 1] = UtilityFunctions.plant_state_to_local_state(plant_state)

    reward_record[:, -1] = state_cost_coeff*(1-plant_state[7])*(1-plant_state[8])
    os.chdir(directory + '/Plant data')
    np.savetxt('PL_state' + str(epi) + '.txt', state_record, fmt='%12.8f')
    np.savetxt('PL_input' + str(epi) + '.txt', input_record, fmt='%12.8f')
    np.savetxt('PL_reward' + str(epi) + '.txt', reward_record, fmt='%12.8f')
    np.savetxt('PL_local_state' + str(epi) + '.txt', local_state_record, fmt='%12.8f')
    np.savetxt('PL_local_input' + str(epi) + '.txt', local_input_record, fmt='%12.8f')
    os.chdir(directory)

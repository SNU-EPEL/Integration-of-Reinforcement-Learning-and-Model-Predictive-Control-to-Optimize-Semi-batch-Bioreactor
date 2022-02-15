# 2021.06.02 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Main script to simulate the bio-reactor with QMPC
# The details are presented in the paper :
# "Integration of Reinforcement Learning and Predictive Control to Optimize Semi-batch bioreactor"

import numpy as np
import tensorflow as tf
import time
import os
import pickle

import UtilityFunctions
from CriticDNN import DNN
from ParaEstimation import ParameterEstimator
from OptimalControl import OptimalController
from PlantSimulator import Simulator

TOTAL_BATCH = 1000
INITIAL_BATCH = 10
BASE_SEED = 100
INI_PERTURB = True
PARA_PERTURB = True
INITIAL_INPUT_PERTURB = 0.0
INPUT_PERTURB = 0.02  # 2% perturb
INITIAL_RANGE = 0.

TRAINING_PERIOD = 100
UPDATING_PERIOD = 5*TRAINING_PERIOD
CORRECTING_PERIOD = 2000
LEARNING_RATE = 0.0002
UPDATING_RATE = 0.02
CORRECTING_RATE = 0.01

BUFFER_MAX = 500000  # About 1000 episodes
INITIAL_BUFFER_COUNT = 0
EPOCH = 500
BATCH_SIZE = 128  # Data batch size

HORIZON_LENGTH = 460
ELEMENT_NUMBER = 1
COLLOCATION_NUMBER = 3
PE_MIN_HORIZON = 1000  # > 460 means no parameter estimation
PE_HORIZON = 15
MPC_MIN_HORIZON = 20
MPC_MAX_HORIZON = 3
MPC_HORIZON = 3
MPC_BOUND = 0.1
PE_RELAX = 8
MPC_RELAX = 3
TRIAL_FLAG = 5
REPEAT_FLAG = 3
MPC_REPEAT_FLAG = 10

total_running_time = time.time()

np.random.seed(BASE_SEED)

# Neural network set-up
nn_trainer = DNN(BUFFER_MAX, LEARNING_RATE, BASE_SEED)
error_history = []
buffer_count = 0

predict_critic = nn_trainer.build_critic()
eval_critic = nn_trainer.build_critic()
eval_critic.set_weights(predict_critic.get_weights())
critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)


# Initial NN weights 
# If the initial weights are used, then please turn off the MC train below
with open('nn_weights.pickle', 'rb') as file:
    nn_initial_weight = pickle.load(file)
predict_critic.set_weights(nn_initial_weight)
eval_critic.set_weights(nn_initial_weight)

'''
# Historical data  
with open('nn_data.pickle', 'rb') as f:
    saved_data = pickle.load(f)
nn_trainer.state_buffer = saved_data[0]
nn_trainer.action_buffer = saved_data[1]
nn_trainer.reward_buffer = saved_data[2]
nn_trainer.next_state_buffer = saved_data[3]
'''

# Create path to save the data
directory = os.getcwd()
if not os.path.exists(directory + '/Plant data'):
    os.mkdir(directory + '/Plant data')

# Delete every file in the path
for file in os.scandir(directory + '/Plant data'):
    os.remove(file.path)

'####################### Initial batch runs ###########################'
for epi in range(INITIAL_BATCH):
    initial_batch_run_time = time.time()
    print('********** Batch number: ', epi, '************')
    # Define the plant
    plant_simulator = Simulator(seed=1000 + epi, ini_perturb=INI_PERTURB, para_perturb=PARA_PERTURB)
    step_time_interval = plant_simulator.plant.time_interval

    # Initial state, input, and value function
    state, _ = plant_simulator.reset()
    u = np.loadtxt('Reference_input.txt')

    # Data record - descaled values
    PL_state_record = np.zeros((plant_simulator.plant.state_dimension, HORIZON_LENGTH + 1))
    PL_input_record = np.zeros((plant_simulator.plant.input_dimension, HORIZON_LENGTH + 1))
    PL_reward_record = np.zeros((1, HORIZON_LENGTH + 1))
    previous_input_record = np.zeros((1, HORIZON_LENGTH + 1))

    for time_step in range(HORIZON_LENGTH):
        PL_state_record[:, time_step] = state
        # Input schedule & perturbation
        u[0:2, time_step] = (1 + INITIAL_RANGE*epi/INITIAL_BATCH)*u[0:2, time_step]
        u[0:2, time_step] = np.multiply(u[0:2, time_step], 1 + INITIAL_INPUT_PERTURB*np.random.uniform(-1, 1, 2))
        next_state, current_input, reward = plant_simulator.step(state, u[:, time_step], time_step)
        PL_input_record[:, time_step] = current_input
        PL_reward_record[:, time_step] = reward
        previous_input_record[:, time_step] = UtilityFunctions.plant_input_to_local_input(current_input)

        # Store the data in buffers
        counter = buffer_count % BUFFER_MAX
        data_tuple = [UtilityFunctions.plant_state_to_local_state(state),
                      UtilityFunctions.plant_input_to_local_input(current_input), reward,
                      UtilityFunctions.plant_state_to_local_state(next_state)]
        nn_trainer.save_data(data_tuple, counter)

        # Next time step
        state = next_state
        buffer_count += 1

    # Terminal step, input is consider as 0
    PL_state_record[:, HORIZON_LENGTH] = state
    next_state, current_input, reward = plant_simulator.step(state, u[:, time_step], HORIZON_LENGTH)
    PL_reward_record[:, HORIZON_LENGTH] = reward

    os.chdir(directory + '/Plant data')
    np.savetxt('PL_state' + str(epi) + '.txt',  PL_state_record, fmt='%12.8f')
    np.savetxt('PL_input' + str(epi) + '.txt',  PL_input_record, fmt='%12.8f')
    np.savetxt('PL_reward' + str(epi) + '.txt',  PL_reward_record, fmt='%12.8f')
    os.chdir(directory)

    print("********** Computation time for a single batch:", time.time() - initial_batch_run_time)

'####################### M-C learning ###########################'
'''
ini_critic = nn_trainer.build_critic2(LEARNING_RATE)
nn_weight, mae, mse, val_mae, val_mse, eval_critic \
    = nn_trainer.train_critic_mc(critic=ini_critic, batch_number=INITIAL_BATCH, epoch=EPOCH, batch_size=BATCH_SIZE)
eval_critic.set_weights(nn_weight)
predict_critic.set_weights(nn_weight)
'''

'####################### Batch runs with controller ###########################'
for epi in range(INITIAL_BATCH, TOTAL_BATCH):
    batch_run_time = time.time()
    print('********** Batch number **********', epi)
    # Define the plant
    plant_simulator = Simulator(seed=BASE_SEED + epi, ini_perturb=INI_PERTURB, para_perturb=PARA_PERTURB)
    step_time_interval = plant_simulator.plant.time_interval

    # Initial state, input, and value function
    state, _ = plant_simulator.reset()
    u = np.loadtxt('Reference_input.txt')

    # Controller Set-up
    nn_weight = predict_critic.get_weights()
    Controller = OptimalController(COLLOCATION_NUMBER, ELEMENT_NUMBER, nn_weight)

    # Estimator Set-up
    para_lower_bound = np.loadtxt('Parameter_lower_bound.txt')
    para_upper_bound = np.loadtxt('Parameter_upper_bound.txt')
    para_initial_guess = np.loadtxt('Parameter_initial_guess.txt')
    estimator = ParameterEstimator(COLLOCATION_NUMBER, ELEMENT_NUMBER, para_lower_bound, para_upper_bound)

    # Data record - descaled values
    PL_state_record = np.zeros((plant_simulator.plant.state_dimension, HORIZON_LENGTH+1))
    PL_input_record = np.zeros((plant_simulator.plant.input_dimension, HORIZON_LENGTH+1))
    PL_reward_record = np.zeros((1, HORIZON_LENGTH + 1))

    PE_state_record = np.zeros((estimator.state_dim, HORIZON_LENGTH+1))
    PE_input_record = np.zeros((estimator.input_dim, HORIZON_LENGTH+1))
    PE_para_record = np.zeros((estimator.parameter_dim, HORIZON_LENGTH+1))
    PE_cost_record = np.zeros((1, HORIZON_LENGTH))
    PE_flag_record = np.zeros((1, HORIZON_LENGTH))

    MPC_state_record = np.zeros((Controller.state_dim, HORIZON_LENGTH+1))
    MPC_input_record = np.zeros((Controller.input_dim, HORIZON_LENGTH+1))
    MPC_cost_record = np.zeros((1, HORIZON_LENGTH))
    MPC_flag_record = np.zeros((1, HORIZON_LENGTH))

    for time_step in range(HORIZON_LENGTH):
        print('***** Batch number ***** ', epi, ' ***** Time step *****', time_step)
        PL_state_record[:, time_step] = state
        PE_state_record[:, time_step] = UtilityFunctions.plant_state_to_local_state(state)
        MPC_state_record[:, time_step] = UtilityFunctions.plant_state_to_local_state(state)

        '####################### PARAMETER ESTIMATION ###########################'
        cost_threshold = 1
        if time_step < 100:
            estimator.weight = np.diag([0, 100, 500, 100, 100])
            cost_threshold = 1000*PE_HORIZON*1e-7
        else:
            estimator.weight = np.diag([0, 200, 500, 100, 100])
            cost_threshold = 10*PE_HORIZON*1e-7

        if time_step > PE_MIN_HORIZON and np.mod(time_step, PE_HORIZON) == 0:
            # data extraction
            PE_s = PE_state_record[:, time_step-PE_HORIZON:time_step+1]
            PE_a = PE_input_record[:, time_step-PE_HORIZON:time_step]
            PE_p = para_initial_guess[:, time_step]

            # parameter estimation
            trial = 0
            repeat_flag = 0

            PE_x_opt, PE_u_opt, PE_p_opt, PE_y_opt, PE_cost = estimator.estimation(
                state_data=PE_s, input_data=PE_a, initial_parameter=PE_p, step_time_interval=step_time_interval,
                time_index=time_step, softing=PE_RELAX)

            print('***** PE ***** Iteration', epi, 'time step', time_step, 'repeat', repeat_flag, 'trial', trial)

            while PE_cost >= cost_threshold:  # PE fail
                repeat_flag += 1
                if repeat_flag == REPEAT_FLAG:
                    repeat_flag = 0
                    trial += 1
                if trial == TRIAL_FLAG:
                    PE_p_opt = PE_para_record[:, time_step - 1]
                    PE_cost = 1
                    cost_threshold = 100
                else:
                    new_guess = np.random.uniform(para_lower_bound[:, time_step], para_upper_bound[:, time_step],
                                                  estimator.parameter_dim)
                    PE_x_opt, PE_u_opt, PE_p_opt, PE_y_opt, PE_cost = estimator.estimation(
                        state_data=PE_s, input_data=PE_a, initial_parameter=PE_p, step_time_interval=step_time_interval,
                        time_index=time_step, softing=PE_RELAX - trial)
                print('***** PE ***** Iteration', epi, 'time step', time_step, 'repeat', repeat_flag, 'trial', trial)
                print('Parameter estimation result, Current cost is: ', PE_cost, 'Threshold', cost_threshold)

            PE_cost_record[:, time_step] = PE_cost
            PE_para_record[:, time_step] = PE_p_opt
            PE_flag_record[:, time_step] = trial
        else:
            if 'PE_p_opt' in locals():
                PE_para_record[:, time_step] = PE_p_opt
            else:
                PE_para_record[:, time_step] = para_initial_guess[:, time_step]

        '####################### MODEL PREDICTIVE CONTROL ###########################'
        # Shrinking horizon
        if HORIZON_LENGTH - time_step < MPC_MAX_HORIZON + 1:
            MPC_HORIZON = HORIZON_LENGTH - time_step
            terminal = True
        else:
            MPC_HORIZON = MPC_MAX_HORIZON
            terminal = False

        if time_step > MPC_MIN_HORIZON:
            # Data extraction
            MPC_s = MPC_state_record[:, time_step]
            # MPC initial input guess (take from previous batch)
            MPC_u_ini = previous_input_record[:, time_step:time_step + MPC_HORIZON]
            # MPC_u_ini = MPC_input_record[:, time_step-1]*np.ones((1, MPC_HORIZON))
            MPC_p = PE_para_record[:, time_step]

            # Control
            MPC_x_opt, MPC_u_opt, MPC_p_opt, MPC_cost, constraint_value = \
                Controller.control(initial_state=MPC_s, initial_input=MPC_u_ini, parameter=MPC_p,
                                   step_time_interval=step_time_interval, n=MPC_HORIZON, bound=MPC_BOUND,
                                   softing=MPC_RELAX, terminal=terminal)

            MPC_flag = 0
            while any(np.array(constraint_value) < -10**(-(MPC_RELAX - 0.3))):  # failed
                MPC_flag = MPC_flag + 1
                if MPC_flag > MPC_REPEAT_FLAG:
                    break
                print('************* MPC fail **************, Trial: ', MPC_flag)
                MPC_s = (1 + 0.02*(np.random.random(5) - 0.5))*MPC_s  # perturb the initial state
                MPC_u_ini = (1 + 0.1*(np.random.random(MPC_HORIZON) - 0.5))*MPC_u_ini  # perturb the input guess
                MPC_x_opt, MPC_u_opt, MPC_p_opt, MPC_cost, constraint_value = \
                    Controller.control(initial_state=MPC_s, initial_input=MPC_u_ini, parameter=MPC_p,
                                       step_time_interval=step_time_interval, n=MPC_HORIZON, bound=MPC_BOUND,
                                       softing=MPC_RELAX, terminal=terminal)
                print('***** MPC Constraint value *****', min(np.array(constraint_value)))

            print('********** MPC inputs', MPC_u_opt[0][0], ' | Reference input ', MPC_u_ini, '**************')
            MPC_flag_record[:, time_step] = MPC_flag
            MPC_cost_record[:, time_step] = MPC_cost
            MPC_input_record[:, time_step] = MPC_u_opt[0][0]

            u[0:2, time_step] = UtilityFunctions.local_input_to_plant_input(MPC_u_opt[0][0], time_step)
            print('***** Episode number ***** ', epi, '*** Time step ***', time_step, '*** MPC cost ***', MPC_cost)

        '####################### Plant Simulation ###########################'
        u[0:2, time_step] = np.multiply(u[0:2, time_step], 1 + INPUT_PERTURB*np.random.uniform(-1, 1, 2))
        next_state, current_input, reward = plant_simulator.step(state, u[:, time_step], time_step)
        PL_input_record[:, time_step] = current_input
        PE_input_record[:, time_step] = UtilityFunctions.plant_input_to_local_input(current_input)
        previous_input_record[:, time_step] = UtilityFunctions.plant_input_to_local_input(current_input)
        PL_reward_record[:, time_step] = reward
        MPC_input_record[:, time_step] = UtilityFunctions.plant_input_to_local_input(current_input)

        '####################### Neural Network training ###########################'
        # store data in buffer
        counter = buffer_count % BUFFER_MAX
        data_tuple = [UtilityFunctions.plant_state_to_local_state(state),
                      UtilityFunctions.plant_input_to_local_input(current_input), reward,
                      UtilityFunctions.plant_state_to_local_state(next_state)]
        nn_trainer.save_data(data_tuple, counter)
        buffer_count += 1

        # training the critic
        if np.mod(buffer_count, TRAINING_PERIOD) == TRAINING_PERIOD - 1:
            batch_size_now = min(BATCH_SIZE, buffer_count)
            indices = np.random.choice(max(INITIAL_BUFFER_COUNT, buffer_count), batch_size_now, replace=False)
            predict_critic, loss = nn_trainer.train_critic(predict_critic, eval_critic, critic_optimizer,
                                                           batch_size_now, indices)
            error_history.append(loss)

        if np.mod(buffer_count, UPDATING_PERIOD) == UPDATING_PERIOD - 1:
            pw = predict_critic.get_weights()
            ew = eval_critic.get_weights()
            for k in range(2*4):  # layer * 2
                ew[k] = (1 - UPDATING_RATE) * ew[k] + UPDATING_RATE * pw[k]
            eval_critic.set_weights(ew)

        # Move to next time step
        state = next_state

    # Terminal step, input is consider as 0
    PL_state_record[:, HORIZON_LENGTH] = state
    PE_state_record[:, HORIZON_LENGTH] = UtilityFunctions.plant_state_to_local_state(state)
    MPC_state_record[:, HORIZON_LENGTH] = UtilityFunctions.plant_state_to_local_state(state)
    next_state, current_input, reward = plant_simulator.step(state, u[:, time_step], HORIZON_LENGTH)
    PL_reward_record[:, HORIZON_LENGTH] = reward

    '####################### M-C learning ###########################'
    if np.mod(epi, CORRECTING_PERIOD) == CORRECTING_PERIOD - 1:
        reg_critic = nn_trainer.build_critic2(LEARNING_RATE)
        reg_critic.set_weights(eval_critic.get_weights())
        nn_weight, mae, mse, val_mae, val_mse, _ \
            = nn_trainer.train_critic_mc(critic=reg_critic, batch_number=epi, epoch=EPOCH, batch_size=BATCH_SIZE)
        rw = reg_critic.get_weights()
        ew = eval_critic.get_weights()
        for k in range(2*4):  # layer * 2
            ew[k] = (1 - CORRECTING_RATE)*ew[k] + CORRECTING_RATE*rw[k]
        eval_critic.set_weights(ew)

    print("********** Computation time for a single batch:", time.time() - batch_run_time)

    # Save the data
    os.chdir(directory + '/Plant data')
    # np.savetxt('train_mae.txt', mae, fmt='%12.8f')
    # np.savetxt('valid_mae.txt', val_mae, fmt='%12.8f')
    np.savetxt('loss_mse.txt', np.array(error_history), fmt='%12.8f')
    np.savetxt('PL_state' + str(epi) + '.txt',  PL_state_record, fmt='%12.8f')
    np.savetxt('PL_input' + str(epi) + '.txt',  PL_input_record, fmt='%12.8f')
    np.savetxt('PL_reward' + str(epi) + '.txt',  PL_reward_record, fmt='%12.8f')
    np.savetxt('PE_state' + str(epi) + '.txt',  PE_state_record, fmt='%12.8f')
    np.savetxt('PE_input' + str(epi) + '.txt',  PE_input_record, fmt='%12.8f')
    np.savetxt('PE_para' + str(epi) + '.txt',   PE_para_record, fmt='%12.8f')
    np.savetxt('PE_cost' + str(epi) + '.txt',   PE_cost_record, fmt='%12.8f')
    np.savetxt('PE_flag' + str(epi) + '.txt',   PE_flag_record, fmt='%12.8f')
    np.savetxt('MPC_state' + str(epi) + '.txt', MPC_state_record, fmt='%12.8f')
    np.savetxt('MPC_input' + str(epi) + '.txt', MPC_input_record, fmt='%12.8f')
    np.savetxt('MPC_cost' + str(epi) + '.txt',  MPC_cost_record, fmt='%12.8f')
    np.savetxt('MPC_flag' + str(epi) + '.txt',  MPC_flag_record, fmt='%12.8f')
    os.chdir(directory)
    with open('nn_train.pickle', 'wb') as f:
        pickle.dump(predict_critic.get_weights(), f, pickle.HIGHEST_PROTOCOL)


save_data = [nn_trainer.state_buffer, nn_trainer.action_buffer, nn_trainer.reward_buffer,
               nn_trainer.next_state_buffer]

with open('nn_data.pickle', 'wb') as f:
    pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

print("********** Computation time for total batches:", time.time() - total_running_time)





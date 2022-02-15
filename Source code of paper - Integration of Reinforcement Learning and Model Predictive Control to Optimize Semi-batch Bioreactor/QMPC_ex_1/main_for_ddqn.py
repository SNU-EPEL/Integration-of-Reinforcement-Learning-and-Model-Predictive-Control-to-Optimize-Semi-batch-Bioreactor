import time
import os
import pickle
import numpy as np
import tensorflow as tf
import UtilityFunctions

from critic_for_ddqn import DNN
from PlantDynamics import PlantSimulator

total_running_time = time.time()

TOTAL_BATCH = 1000
INITIAL_BATCH = 10
BASE_SEED = 1000
STATE_PERTURB = False
PARA_PERTURB = False

LEARNING_PERIOD = 2
UPDATING_PERIOD = 5*LEARNING_PERIOD
LEARNING_RATE = 0.01
UPDATING_RATE = 0.02

BUFFER_MAX = 500000
INITIAL_BUFFER_COUNT = 0
EPOCH = 10
BATCH_SIZE = 32  # Data batch size

EXPLORATION_RATE = 0.3
ACTION_NUMBER1 = 10
ACTION_NUMBER2 = 10

HORIZON_LENGTH = 20
COLLOCATION_NUMBER = 3
ELEMENT_NUMBER = 1

NODE_NUMBER = [16, 4, 1]  # [4, 4, 1]


# Create path to save the data
directory = os.getcwd()
if not os.path.exists(directory + '/Plant data ddqn'):
    os.mkdir(directory + '/Plant data ddqn')

# Delete every file in the path
for file in os.scandir(directory + '/Plant data ddqn'):
    os.remove(file.path)


with open('nn_train_ini.pickle', 'rb') as f:
    nn_initial_weight = pickle.load(f)

'''
with open('nn_data.pickle', 'rb') as f:
    all_data = pickle.load(f)
'''

np.random.seed(BASE_SEED)

# Plant set-up
plant = PlantSimulator(seed=BASE_SEED, state_perturb=STATE_PERTURB, parameter_perturb=PARA_PERTURB)


# initialize replay memory D to capacity N
# Neural network set-up
nn_trainer = DNN(state_dim=plant.state_dimension, input_dim=plant.input_dimension, node_number=NODE_NUMBER,
                 buffer_size=BUFFER_MAX, learning_rate=LEARNING_RATE, terminal_cost=plant.terminal_cost, seed=BASE_SEED,
                 state_min=plant.xmin, state_max=plant.xmax, input_min=plant.umin, input_max=plant.umax)

buffer_count = 0
'''
# Data saving if any... 
DNN.state_buffer = all_data[0]
DNN.action_buffer = all_data[1]
DNN.reward_buffer = all_data[2]
DNN.next_state_buffer = all_data[3]
'''

# Initial NN weights
# If the initial weights are used, then please turn off the MC train below
predict_critic = nn_trainer.build_critic()
eval_critic = nn_trainer.build_critic()
predict_critic.set_weights(nn_initial_weight)
eval_critic.set_weights(nn_initial_weight)
eval_critic.set_weights(predict_critic.get_weights())

dddd = np.zeros(6)
for k in range(6):
    dddd[k] = np.sum(predict_critic.get_weights()[k])

critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
error_history = []
cost_history = []

for epi in range(TOTAL_BATCH):
    batch_run_time = time.time()
    print("Batch :", epi)
    if epi < INITIAL_BATCH:
        plant.seed = BASE_SEED + TOTAL_BATCH + epi
    else:
        plant.seed = BASE_SEED + epi

    # initialize sequence
    s, _ = plant.reset()
    u = 6*np.random.random((2, HORIZON_LENGTH+1))
    #u = np.array([[2.0, 10/3, 10/3, 10/3, 10/3, 10/3, 10/3, 10/3, 10/3, 10/3, 16/3],
    #              [0.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0,  6.0]])
    p = np.array([0.2])

    # Data record - descaled values
    PL_state_record = np.zeros((plant.state_dimension, HORIZON_LENGTH + 1))
    PL_input_record = np.zeros((plant.input_dimension, HORIZON_LENGTH + 1))
    PL_reward_record = np.zeros((1, HORIZON_LENGTH + 1))

    DQN_state_record = np.zeros((plant.state_dimension, HORIZON_LENGTH+1))
    DQN_input_record = np.zeros((plant.input_dimension, HORIZON_LENGTH+1))

    for t in range(HORIZON_LENGTH):
        # print("Batch, time Step is :", epi, t)
        PL_state_record[:, t] = s
        DQN_state_record[:, t] = s

        if epi > INITIAL_BATCH - 1:
            DQN_s = DQN_state_record[:, t]
            DQN_s = UtilityFunctions.scale(DQN_s, plant.xmin, plant.xmax)
            if np.random.random(1) < EXPLORATION_RATE*(1 - epi/TOTAL_BATCH) - 0.1:             ##############################
                # if eeeee < EXPLORATION_RATE:
                #DQN_input = UtilityFunctions.scale(DQN_input_record[:, t-1], UMIN, UMAX) \
                #            + np.random.choice(np.linspace(-0.1, 0.1, num=3), 1)
                a1 = np.random.choice(np.linspace(0, 1, num=ACTION_NUMBER1), 1)
                a2 = np.random.choice(np.linspace(0, 1, num=ACTION_NUMBER2), 1)
                DQN_input = np.array([a1[0], a2[0]])
            else:
                DQN_input, _ = nn_trainer.q_minimization(critic=predict_critic, state=np.array([DQN_s]),
                                                         action_number1=ACTION_NUMBER1, action_number2=ACTION_NUMBER2,
                                                         ref_act=np.zeros(2))
            # print('DQN_Input', DQN_input)
            DQN_input = np.clip(DQN_input, 0, 1)
            DQN_input_record[:, t] = UtilityFunctions.descale(DQN_input, plant.umin, plant.umax)
            # print('input', DQN_input_record[:, t])
            u[:, t] = DQN_input_record[:, t]

        else:
            111
            #a1 = np.random.choice(np.linspace(0, 1, num=ACTION_NUMBER1), 1)
            #a2 = np.random.choice(np.linspace(0, 1, num=ACTION_NUMBER2), 1)
            #u[:, t] = 6 * np.array([a1[0], a2[0]])

        # store transition in D
        # print('h3', s, u[:, t], p)

        next_state, current_input, reward = plant.step(s, u[:, t], p, terminal_flag=False)
        PL_input_record[:, t] = current_input
        PL_reward_record[:, t] = reward

        # store data in buffer
        counter = buffer_count % BUFFER_MAX
        batch_size_now = min(BATCH_SIZE, buffer_count)
        nn_trainer.state_buffer[counter, :] = UtilityFunctions.scale(s, plant.xmin, plant.xmax)
        nn_trainer.action_buffer[counter, :] = UtilityFunctions.scale(current_input, plant.umin, plant.umax)
        nn_trainer.reward_buffer[counter, :] = reward
        nn_trainer.next_state_buffer[counter, :] = UtilityFunctions.scale(next_state, plant.xmin, plant.xmax)

        if np.mod(buffer_count, LEARNING_PERIOD) == LEARNING_PERIOD - 1:

            # sample minibatch of transitions from D
            # indices = np.random.choice(buffer_count, batch_size_now, replace=False)
            indices = np.random.choice(max(0, buffer_count), batch_size_now, replace=False)
            state_batch = nn_trainer.state_buffer[indices, :]
            action_batch = nn_trainer.action_buffer[indices, :]
            reward_batch = nn_trainer.reward_buffer[indices, :]
            next_state_batch = nn_trainer.next_state_buffer[indices, :]

            # calculate target value using eval_critic
            y = np.zeros((batch_size_now, 1))
            for k in range(batch_size_now):
                ss = np.array([next_state_batch[k, :]])
                if abs(1.0 - ss[0][0]) < 0.0001:  # #########
                    nnv = nn_trainer._scaled_terminal_cost(ss[0], 0)
                    # print('terminal?', ss[0][0], nnv)
                else:
                    act, nnv = nn_trainer.q_minimization(eval_critic, ss, ACTION_NUMBER1, ACTION_NUMBER2,
                                                         action_batch[k, :])
                y[k, :] = reward_batch[k, :] + np.clip(nnv, 0, 2)  # No consideration of terminal cost


            # print('y', y)
            # perform gradient descent w.r.t. pred_critic
            with tf.GradientTape() as tape:
                sa_input = np.hstack([state_batch, action_batch])
                q_value = predict_critic(sa_input, training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(y - q_value))
                # print('loss', critic_loss)
            critic_grad = tape.gradient(critic_loss, predict_critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grad, predict_critic.trainable_variables))
            error_history.append(critic_loss)
            ddd = np.zeros(6)
            for k in range(6):
                ddd[k] = np.sum(predict_critic.get_weights()[k])
            #print('para:',
            #      np.sum(predict_critic.get_weights()[0]),
            #      np.sum(predict_critic.get_weights()[1]),
            #      np.sum(predict_critic.get_weights()[2]),
            #      np.sum(predict_critic.get_weights()[3]),
            #      np.sum(predict_critic.get_weights()[4]),
            #      np.sum(predict_critic.get_weights()[5]))
            # print('del_weight:', np.sum(np.abs(ddd - dddd)))
            with open('nn_tra_ddqn.pickle', 'wb') as f:
                pickle.dump(predict_critic.get_weights(), f, pickle.HIGHEST_PROTOCOL)


        if np.mod(buffer_count, UPDATING_PERIOD) == UPDATING_PERIOD - 1:
            pw = predict_critic.get_weights()
            ew = eval_critic.get_weights()
            for k in range(int(2*len(NODE_NUMBER))):  # layer * 2
                ew[k] = (1 - UPDATING_RATE) * ew[k] + UPDATING_RATE * pw[k]
            eval_critic.set_weights(ew)
        buffer_count += 1
        s = next_state

    # Terminal step, input is consider as 0
    PL_state_record[:, HORIZON_LENGTH] = s
    next_state, current_input, reward = plant.step(s, u[:, t], p, terminal_flag=True)
    PL_reward_record[:, HORIZON_LENGTH] = reward
    print("***reward:", reward, " ******* Computation time for a single batch:", time.time() - batch_run_time)
    # Save the data
    os.chdir(directory + '/Plant data ddqn')
    np.savetxt('PL_state' + str(epi) + '.txt',  PL_state_record, fmt='%12.8f')
    np.savetxt('PL_input' + str(epi) + '.txt',  PL_input_record, fmt='%12.8f')
    np.savetxt('PL_reward' + str(epi) + '.txt',  PL_reward_record, fmt='%12.8f')
    np.savetxt('DQN_state' + str(epi) + '.txt', DQN_state_record, fmt='%12.8f')
    np.savetxt('DQN_input' + str(epi) + '.txt', DQN_input_record, fmt='%12.8f')
    np.savetxt('loss_mse.txt', np.array(error_history), fmt='%12.8f')
    os.chdir(directory)

    with open('nn_tra_ddqn.pickle', 'wb') as f:
        pickle.dump(predict_critic.get_weights(), f, pickle.HIGHEST_PROTOCOL)

all_data = [nn_trainer.state_buffer, nn_trainer.action_buffer, nn_trainer.reward_buffer, nn_trainer.next_state_buffer]

with open('nn_data_ddqn.pickle', 'wb') as f:
    pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

print("********** Computation time for total batches:", time.time() - total_running_time)

import time
import os
import pickle
import numpy as np
import tensorflow as tf
import UtilityFunctions

from OUNoise import OUActionNoise
from critic_for_ddpg import DNN
from PlantDynamics import PlantSimulator

total_running_time = time.time()

TOTAL_BATCH = 1000
INITIAL_BATCH = 10
BASE_SEED = 1000
STATE_PERTURB = False
PARA_PERTURB = False

BUFFER_MAX = 500000
INITIAL_BUFFER_COUNT = 0
EPOCH = 10
BATCH_SIZE = 8  # Data batch size

LEARNING_PERIOD = 1
UPDATING_PERIOD = 5*LEARNING_PERIOD
LEARNING_RATE = 0.001
ACTOR_LEARNING_RATE = 0.012
UPDATING_RATE = 0.02

EXPLORATION_RATE = 0.3
STD_DEV = 0.

HORIZON_LENGTH = 20
COLLOCATION_NUMBER = 3
ELEMENT_NUMBER = 1

CRITIC_NODE_NUMBER = [16, 4, 1]
ACTOR_NODE_NUMBER = [8, 4, 2]

# Create path to save the data
directory = os.getcwd()
if not os.path.exists(directory + '/Plant data ddpg'):
    os.mkdir(directory + '/Plant data ddpg')

# Delete every file in the path
for file in os.scandir(directory + '/Plant data ddpg'):
    os.remove(file.path)

with open('nn_train_ini.pickle', 'rb') as f:
    nn_initial_weight = pickle.load(f)


with open('nn_act_ini_ddpg.pickle', 'rb') as f:
    nn_initial_act_weight = pickle.load(f)

'''
with open('nn_data.pickle', 'rb') as f:
    all_data = pickle.load(f)
'''

# Plant set-up
plant = PlantSimulator(seed=BASE_SEED, state_perturb=STATE_PERTURB, parameter_perturb=PARA_PERTURB)

# initialize replay memory D to capacity N
# Neural network set-up
nn_trainer = DNN(state_dim=plant.state_dimension, input_dim=plant.input_dimension,
                 critic_node_number=CRITIC_NODE_NUMBER, actor_node_number=ACTOR_NODE_NUMBER,
                 buffer_size=BUFFER_MAX, learning_rate=LEARNING_RATE, terminal_cost=plant.terminal_cost, seed=BASE_SEED,
                 state_min=plant.xmin, state_max=plant.xmax, input_min=plant.umin, input_max=plant.umax)

buffer_count = 0

# Initialize actors and critics
predict_critic = nn_trainer.build_critic()
eval_critic = nn_trainer.build_critic()
predict_critic.set_weights(nn_initial_weight)
eval_critic.set_weights(nn_initial_weight)
critic_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
critic_error_history = []

behavior_actor = nn_trainer.build_actor()
target_actor = nn_trainer.build_actor()
#behavior_actor.set_weights(nn_initial_act_weight)
#target_actor.set_weights(nn_initial_act_weight)
target_actor.set_weights(behavior_actor.get_weights())
actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LEARNING_RATE)
actor_error_history = []
cost_history = []


for epi in range(TOTAL_BATCH):
    batch_run_time = time.time()
    print("Batch :", epi)
    if epi < INITIAL_BATCH:
        plant.seed = BASE_SEED + TOTAL_BATCH + epi
    else:
        plant.seed = BASE_SEED + epi

    ou_noise = OUActionNoise(mean=np.zeros(1), std_dev=float(STD_DEV*(1 - epi/TOTAL_BATCH)) * np.ones(1))  ###########

    # initialize sequence
    s, _ = plant.reset()
    u = 6 * np.random.random((2, HORIZON_LENGTH + 1))
    p = np.array([0.2])

    # Data record - descaled values
    PL_state_record = np.zeros((plant.state_dimension, HORIZON_LENGTH + 1))
    PL_input_record = np.zeros((plant.input_dimension, HORIZON_LENGTH + 1))
    PL_reward_record = np.zeros((1, HORIZON_LENGTH + 1))

    DDPG_state_record = np.zeros((plant.state_dimension, HORIZON_LENGTH+1))
    DDPG_input_record = np.zeros((plant.input_dimension, HORIZON_LENGTH+1))

    for t in range(HORIZON_LENGTH):
        # print("Batch, time Step is :", epi, t)
        PL_state_record[:, t] = s
        DDPG_state_record[:, t] = s

        #if epi < INITIAL_BATCH:
        #    u = np.array([[2.0, 2.0, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3,
        #                   10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 10 / 3, 16 / 3, 16 / 3],
        #                  [0.0, 0.0, 16.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        #                   6.0, 6.0, 6.0]])
        #    u = np.clip(u*(1 + 0.05*np.random.randn(1)), 0, 6)

        if epi > INITIAL_BATCH - 1:
            if np.random.random(1) < EXPLORATION_RATE*(1 - epi/TOTAL_BATCH) - 0.1:             ##############################
                DDPG_input_record[:, t] = 6*np.random.random(2)
                u[:, t] = DDPG_input_record[:, t]
            else:
                DDPG_s = DDPG_state_record[:, t]
                DDPG_s = UtilityFunctions.scale(DDPG_s, plant.xmin, plant.xmax)
                DDPG_input = nn_trainer.policy(DDPG_s, behavior_actor, ou_noise)
                # print('t, a', t, DQN_input)
                DDPG_input = np.clip(DDPG_input, 0, 1)
                DDPG_input_record[:, t] = UtilityFunctions.descale(DDPG_input, plant.umin, plant.umax)
                # print('input', DQN_input_record[:, t])
                u[:, t] = DDPG_input_record[:, t]

        # store transition in D
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


        # sample minibatch of transitions from D
        # indices = np.random.choice(buffer_count, batch_size_now, replace=False)
        indices = np.random.choice(max(0, buffer_count), batch_size_now, replace=False)
        state_batch = nn_trainer.state_buffer[indices, :]
        action_batch = nn_trainer.action_buffer[indices, :]
        reward_batch = nn_trainer.reward_buffer[indices, :]
        next_state_batch = nn_trainer.next_state_buffer[indices, :]
        # print(buffer_count, indices, state_batch)

        # Actor update for every step
        with tf.GradientTape() as tape:
            a = behavior_actor(state_batch, training=True)
            # print(state_batch, a)
            sa = tf.concat([state_batch, a], 1)  # need to use tf.concat instead of numpy for gradient calculation.
            critic_value = predict_critic(sa, training=True)
            actor_loss = tf.math.reduce_mean(critic_value)  # minimize Q(S,A)
        actor_grad = tape.gradient(actor_loss, behavior_actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, behavior_actor.trainable_variables))
        actor_error_history.append(actor_loss)

        if np.mod(buffer_count, LEARNING_PERIOD) == LEARNING_PERIOD - 1:

            # calculate target value using behavior_critic
            y = np.zeros((batch_size_now, 1))
            for k in range(batch_size_now):
                ss = np.array([next_state_batch[k, :]])
                if abs(1.0 - ss[0][0]) < 0.0001:  # #########
                    nnv = nn_trainer.terminal_cost(ss[0], 0)
                    # print('terminal?', ss[0][0], nnv)
                else:
                    act = target_actor(np.array([next_state_batch[k, :]]), training=True)
                    sa_plus = tf.concat([np.array([next_state_batch[k, :]]), act], 1)
                    nnv = eval_critic(sa_plus, training=True)
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
            critic_error_history.append(critic_loss)

        if np.mod(buffer_count, UPDATING_PERIOD) == UPDATING_PERIOD - 1:
            pw = predict_critic.get_weights()
            ew = eval_critic.get_weights()
            apw = behavior_actor.get_weights()
            aew = target_actor.get_weights()
            for k in range(int(2*len(CRITIC_NODE_NUMBER))):  # layer * 2
                ew[k] = (1 - UPDATING_RATE) * ew[k] + UPDATING_RATE * pw[k]
            eval_critic.set_weights(ew)
            for k in range(int(2*len(ACTOR_NODE_NUMBER))):
                aew[k] = (1 - UPDATING_RATE) * aew[k] + UPDATING_RATE * apw[k]
            target_actor.set_weights(aew)

        buffer_count += 1
        s = next_state

    # Terminal step, input is consider as 0
    PL_state_record[:, HORIZON_LENGTH] = s
    next_state, current_input, reward = plant.step(s, u[:, t], p, terminal_flag=True)
    PL_reward_record[:, HORIZON_LENGTH] = reward
    print("***reward:", reward, " ******* Computation time for a single batch:", time.time() - batch_run_time)
    # Save the data
    os.chdir(directory + '/Plant data ddpg')
    np.savetxt('PL_state' + str(epi) + '.txt', PL_state_record, fmt='%12.8f')
    np.savetxt('PL_input' + str(epi) + '.txt', PL_input_record, fmt='%12.8f')
    np.savetxt('PL_reward' + str(epi) + '.txt', PL_reward_record, fmt='%12.8f')
    np.savetxt('DDPG_state' + str(epi) + '.txt', DDPG_state_record, fmt='%12.8f')
    np.savetxt('DDPG_input' + str(epi) + '.txt', DDPG_input_record, fmt='%12.8f')
    np.savetxt('loss_mse.txt', np.array(critic_error_history), fmt='%12.8f')
    np.savetxt('actor_loss_mse.txt', np.array(actor_error_history), fmt='%12.8f')
    os.chdir(directory)

    with open('nn_tra_ddpg.pickle', 'wb') as f:
        pickle.dump(predict_critic.get_weights(), f, pickle.HIGHEST_PROTOCOL)

    with open('nn_act_ddpg.pickle', 'wb') as f:
        pickle.dump(behavior_actor.get_weights(), f, pickle.HIGHEST_PROTOCOL)

all_data = [nn_trainer.state_buffer, nn_trainer.action_buffer, nn_trainer.reward_buffer, nn_trainer.next_state_buffer]

with open('nn_data_ddpg.pickle', 'wb') as f:
    pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

print("********** Computation time for total batches:", time.time() - total_running_time)
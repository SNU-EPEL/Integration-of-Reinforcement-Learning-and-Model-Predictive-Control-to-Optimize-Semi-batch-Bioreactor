import os
import random
import pickle
import UtilityFunctions


import numpy as np
import casadi as ca
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow import keras
from tensorflow.keras import layers


NODE_NUMBER = [16, 16, 8, 1]
VALIDATION_RATIO = 0.2
MIN_VALUE = 0
MAX_VALUE = 2

ACTION_BOUND = 0.1
TIME_STEP = 460
STATE_COST_COEFFICIENT = 0.5
STATE_COST_COEFFICIENT2 = 0.5


def custom_activation(x):
    tt = tf.math.log(x ** 2 + 1)
    return tt


get_custom_objects().update({'custom': Activation(custom_activation)})


class DNN(object):
    def __init__(self, buffer_size, learning_rate, seed):
        self.xmin = np.array([0., 0., 0.0001, 0., 0.])
        self.xmax = np.array([230., 150., 25., 100., 110000.])
        self.umin = np.array([10.])
        self.umax = np.array([240.])
        self.xdim = 5
        self.udim = 1
        self.buffer_size = buffer_size

        self.state_buffer = np.zeros((self.buffer_size, self.xdim))
        self.action_buffer = np.zeros((self.buffer_size, self.udim))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, self.xdim))

        self.learning_rate = learning_rate
        self.seed = seed

    def save_data(self, data_tuple, count):
        # descaled in
        # data = [S, A, R, S+]
        self.state_buffer[count, :] = UtilityFunctions.scale(data_tuple[0], self.xmin, self.xmax)
        self.action_buffer[count, :] = UtilityFunctions.scale(data_tuple[1], self.umin, self.umax)
        self.reward_buffer[count, :] = data_tuple[2]
        self.next_state_buffer[count, :] = UtilityFunctions.scale(data_tuple[3], self.xmin, self.xmax)

    def build_critic(self):
        critic_model = keras.Sequential(
            [
                # layers.Dropout(0.2, seed=100, input_shape=(self.xdim + self.udim,),),
                layers.Dense(NODE_NUMBER[0], activation='custom', input_shape=(self.xdim + self.udim,), name='layer1'),
                layers.Dense(NODE_NUMBER[1], activation='custom', name='layer2'),
                layers.Dense(NODE_NUMBER[2], activation='custom', name='layer3'),
                layers.Dense(NODE_NUMBER[3], activation='custom', name='layer4'),
            ]
        )
        return critic_model

    def build_critic2(self, learning_rate):
        critic_model = keras.Sequential(
            [
                # layers.Dropout(0.2, seed=self.seed, input_shape=(self.xdim + self.udim,), ),
                layers.Dense(NODE_NUMBER[0], activation='custom', input_shape=(self.xdim + self.udim,), name='layer1'),
                layers.Dense(NODE_NUMBER[1], activation='custom', name='layer2'),
                layers.Dense(NODE_NUMBER[2], activation='custom', name='layer3'),
                layers.Dense(NODE_NUMBER[3], activation='custom', name='layer4'),
            ]
        )
        critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        critic_model.compile(
            loss='mse',
            optimizer=critic_optimizer,
            metrics=['mae', 'mse']
        )
        return critic_model

    def q_minimization(self, critic, state, action):
        # scaled in, scaled out
        # state : n vector
        # opt_action : nu, 1 matrix

        model_state = ca.SX.sym('x', self.xdim)
        model_input = ca.SX.sym('u', self.udim)
        model_state_input = ca.vertcat(model_state, model_input)
        nn_sym = self.neural_network_casadi(model_state_input, critic.get_weights())
        nn_fcn = ca.Function('nn_func', [model_state, model_input], [nn_sym], ['x', 'u'], ['nn'])

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []

        xx = ca.MX.sym('X', self.xdim)
        w.append(xx)
        lbw = np.append(lbw, np.zeros((self.xdim, 1)))
        ubw = np.append(ubw, np.ones((self.xdim, 1)))
        w0 = np.append(w0, np.zeros((self.xdim, 1)))
        g.append(xx - state)
        lbg = np.append(lbg, np.zeros((self.xdim, 1)))
        ubg = np.append(ubg, np.zeros((self.xdim, 1)))

        uu = ca.MX.sym('U', self.udim)
        w.append(uu)
        lbw = np.append(lbw, np.zeros((self.udim, 1)))
        ubw = np.append(ubw, np.ones((self.udim, 1)))
        w0 = np.append(w0, np.ones((self.udim, 1)))
        g.append(uu - action)
        lbg = np.append(lbg, -ACTION_BOUND*np.ones((self.udim, 1)))
        ubg = np.append(ubg, ACTION_BOUND*np.ones((self.udim, 1)))

        cost = nn_fcn(xx, uu)

        w = ca.vertcat(*w)
        g = ca.vertcat(*g)

        # Create an NLP solver
        prob = {'f': cost, 'x': w, 'g': g}

        # "linear_solver": "ma27"
        # opts = {'ipopt': {'print_level': 0}}  # 'ipopt': {'print_level': 0}
        opts = {'print_time': False, "ipopt": {'print_level': 0}}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        opt_value = np.array(sol['f'])
        xu_value = np.array(sol['x'])
        opt_action = xu_value[self.xdim:, :]

        # print('NN min', opt_action, opt_value)

        return opt_action, opt_value

    def train_critic(self, predict_critic, eval_critic, critic_optimizer, batch_size_now, indices):
        state_batch = self.state_buffer[indices, :]
        action_batch = self.action_buffer[indices, :]
        reward_batch = self.reward_buffer[indices, :]
        next_state_batch = self.next_state_buffer[indices, :]

        # calculate target value using eval_critic
        y = np.zeros((batch_size_now, 1))
        for k in range(batch_size_now):
            next_state = next_state_batch[k, :]
            action = action_batch[k, :]
            if abs(1.0 - next_state[0]) < 0.0001:  # detecting the terminal
                value = self.terminal_value(next_state)
                # print('terminal?', ss[0][0], value)
            else:
                act, value = self.q_minimization(eval_critic, next_state, action)
            y[k, :] = reward_batch[k, :] + np.clip(value, MIN_VALUE, MAX_VALUE)
        # print('y', y)

        # perform gradient descent w.r.t. predict_critic
        with tf.GradientTape() as tape:
            sa_input = np.hstack([state_batch, action_batch])
            q_value = predict_critic(sa_input, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - q_value))
            # print('loss', critic_loss)
        critic_grad = tape.gradient(critic_loss, predict_critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, predict_critic.trainable_variables))
        return predict_critic, critic_loss

    # Pre-training the action-value function by MC method
    def train_critic_mc(self, critic, batch_number, epoch, batch_size):

        train_buffer = []
        valid_buffer = []

        index = np.arange(batch_number)
        np.random.shuffle(index)
        slice_var = int(VALIDATION_RATIO * len(index))
        train_index = index[slice_var:]

        directory = os.getcwd()
        os.chdir(directory + '/Plant data')
        for k in range(batch_number):
            state_data = UtilityFunctions.plant_state_to_local_state(np.loadtxt("PL_state" + str(k) + ".txt"))
            input_data = UtilityFunctions.plant_input_to_local_input(np.loadtxt("PL_input" + str(k) + ".txt"))
            reward_data = np.loadtxt("PL_reward" + str(k) + ".txt")
            scaled_input_data = UtilityFunctions.scale(input_data, self.umin, self.umax)
            scaled_state_data = np.zeros_like(state_data)
            for kk in range(TIME_STEP + 1):
                scaled_state_data[:, kk] = UtilityFunctions.scale(state_data[:, kk], self.xmin, self.xmax)
            for kk in range(TIME_STEP):
                data_tuple = [scaled_state_data[:, kk].tolist(), scaled_input_data[kk], reward_data[kk],
                              scaled_state_data[:, kk + 1].tolist()]
                if k in train_index:
                    train_buffer.append(data_tuple)
                else:
                    valid_buffer.append(data_tuple)
        os.chdir(directory)

        train_ex = np.zeros((len(train_buffer), self.xdim + self.udim))
        train_la = np.zeros((len(train_buffer), 1))
        for k in range(len(train_buffer)//TIME_STEP):  # For each batch
            for kk in range(TIME_STEP):  # For each time step
                train_ex[TIME_STEP*k + kk, :self.xdim] = train_buffer[TIME_STEP*k + kk][0]
                train_ex[TIME_STEP*k + kk, self.xdim:self.xdim + self.udim] = train_buffer[TIME_STEP*k + kk][1]
                mc_value = 0
                for kkk in range(kk, TIME_STEP):
                    mc_value += train_buffer[TIME_STEP*k + kkk][2]
                mc_value += self.terminal_value(train_buffer[TIME_STEP*(k+1) - 1][3])
                train_la[TIME_STEP*k + kk, :] = mc_value
        print('train_la', train_la)

        valid_ex = np.zeros((len(valid_buffer), self.xdim + self.udim))
        valid_la = np.zeros((len(valid_buffer), 1))
        for k in range(len(valid_buffer)//TIME_STEP):
            for kk in range(TIME_STEP):
                valid_ex[TIME_STEP*k + kk, :self.xdim] = valid_buffer[TIME_STEP*k + kk][0]
                valid_ex[TIME_STEP*k + kk, self.xdim:self.xdim + self.udim] = valid_buffer[TIME_STEP*k + kk][1]
                mc_value = 0
                for kkk in range(kk, TIME_STEP):
                    mc_value += valid_buffer[TIME_STEP*k + kkk][2]
                mc_value += self.terminal_value(valid_buffer[TIME_STEP*(k+1) - 1][3])
                valid_la[TIME_STEP*k + kk, :] = mc_value

        history = critic.fit(train_ex, train_la, epochs=epoch, batch_size=batch_size,
                             validation_data=(valid_ex, valid_la))
        mae = np.array(history.history["mae"])
        mse = np.array(history.history["mse"])
        val_mae = np.array(history.history["val_mae"])
        val_mse = np.array(history.history["val_mse"])
        nn_weight = critic.get_weights()

        return nn_weight, mae, mse, val_mae, val_mse, critic

    @staticmethod
    def neural_network_casadi(x, nn_weight):
        out = x
        for k in range(len(NODE_NUMBER)):
            out = activation(ca.transpose(ca.mtimes(ca.transpose(out), nn_weight[2*k])) + nn_weight[2*k+1])
        return out

    @ staticmethod
    def terminal_value(state):
        terminal_value = STATE_COST_COEFFICIENT*(1 - state[3] * state[4]) + STATE_COST_COEFFICIENT2*(1 - state[3])
        return terminal_value


def activation(x):
    x_len, _ = x.shape
    for k in range(x_len):
        x[k, :] = ca.log(1 + x[k, :]*x[k, :])
        y = x
    return y

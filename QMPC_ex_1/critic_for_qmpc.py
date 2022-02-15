import os
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


# VALIDATION_RATIO = 0.2
# ACTION_BOUND = 0.05
TIME_STEP = 10  # Change later ##########################

# @ staticmethod
# def terminal_value(state):
#    terminal_value = 0.5*(1 - state[3] * state[4]) + 0.5*(1 - state[3])
#    return terminal_value


def custom_activation(x):
    tt = tf.math.log(x ** 2 + 1)
    return tt


get_custom_objects().update({'custom': Activation(custom_activation)})


class DNN(object):
    def __init__(self, state_dim, input_dim, node_number, buffer_size, learning_rate, terminal_cost, seed=100,
                 state_min=None, state_max=None,
                 input_min=None, input_max=None):

        self.node_number = node_number
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.terminal_cost = terminal_cost
        self.seed = seed

        # Default values
        if state_min is None:
            state_min = -10**10*np.ones(self.state_dim)
        if state_max is None:
            state_max = 10**10*np.ones(self.state_dim)
        if input_min is None:
            input_min = -10**10*np.ones(self.input_dim)
        if input_max is None:
            input_max = 10**10*np.ones(self.input_dim)
        self.state_min = state_min
        self.state_max = state_max
        self.input_min = input_min
        self.input_max = input_max

        self.state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = np.zeros((self.buffer_size, self.input_dim))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim))

        # For numerical issues
        self.value_min = 0
        self.value_max = 10

    def save_data(self, data_list, count):
        # descaled in
        # data = [S, A, R, S+] list of vectors
        self.state_buffer[count, :] = UtilityFunctions.scale(data_list[0], self.state_min, self.state_max)
        self.action_buffer[count, :] = UtilityFunctions.scale(data_list[1], self.input_min, self.input_max)
        self.reward_buffer[count, :] = data_list[2]
        self.next_state_buffer[count, :] = UtilityFunctions.scale(data_list[3], self.state_min, self.state_max)

    def build_critic(self):
        stacked_layers = []
        for idx, node_num in enumerate(self.node_number):
            if idx is 0:
                stacked_layers.append(layers.Dense(node_num, activation='custom',
                                      input_shape=(self.state_dim + self.input_dim,), name='layer' + str(idx)))
            else:
                stacked_layers.append(layers.Dense(node_num, activation='custom', name='layer' + str(idx)))
        critic_model = keras.Sequential(stacked_layers)
        return critic_model

    def _scaled_terminal_cost(self, state, action):
        return self.terminal_cost(UtilityFunctions.descale(state, self.state_min, self.state_max),
                                  UtilityFunctions.descale(action, self.input_min, self.input_max))

    def _q_minimization(self, critic, state, action, action_bound):
        # scaled in, scaled out
        # state : n vector
        # opt_action : nu, 1 matrix

        state_sym = ca.SX.sym('x', self.state_dim)
        input_sym = ca.SX.sym('u', self.input_dim)
        state_input_sym = ca.vertcat(state_sym, input_sym)
        nn_sym = self._neural_network_casadi(state_input_sym, critic.get_weights())
        nn_fcn = ca.Function('nn_func', [state_sym, input_sym], [nn_sym], ['x', 'u'], ['nn'])

        # Start with an empty NLP
        w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []

        xx = ca.MX.sym('X', self.state_dim)
        w.append(xx)
        lbw = np.append(lbw, np.zeros((self.state_dim, 1)))
        ubw = np.append(ubw, np.ones((self.state_dim, 1)))
        w0 = np.append(w0, np.zeros((self.state_dim, 1)))
        g.append(xx - state)
        lbg = np.append(lbg, np.zeros((self.state_dim, 1)))
        ubg = np.append(ubg, np.zeros((self.state_dim, 1)))

        uu = ca.MX.sym('U', self.input_dim)
        w.append(uu)
        lbw = np.append(lbw, np.zeros((self.input_dim, 1)))
        ubw = np.append(ubw, np.ones((self.input_dim, 1)))
        # w0 = np.append(w0, np.ones((self.input_dim, 1)))
        w0 = np.append(w0, action)
        g.append(uu - action)
        lbg = np.append(lbg, -action_bound*np.ones((self.input_dim, 1)))
        ubg = np.append(ubg, action_bound*np.ones((self.input_dim, 1)))

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
        opt_action = xu_value[self.state_dim:, :]

        # print('NN min', opt_action, opt_value)

        return opt_action, opt_value

    def train_critic(self, action_bound, predict_critic, eval_critic, critic_optimizer, batch_size_now, indices):
        state_batch = self.state_buffer[indices, :]
        action_batch = self.action_buffer[indices, :]
        reward_batch = self.reward_buffer[indices, :]
        next_state_batch = self.next_state_buffer[indices, :]

        # calculate target value using eval_critic
        y = np.zeros((batch_size_now, 1))
        for k in range(batch_size_now):
            next_state = next_state_batch[k, :]
            action = action_batch[k, :]
            # print('next state', next_state)
            if abs(1.0 - next_state[0]) < 0.0001:  # detecting the terminal # may change later ...
                value = self._scaled_terminal_cost(next_state, 0)  # arbitrary input
            else:
                act, value = self._q_minimization(eval_critic, next_state, action, action_bound)
                # print('q', act, value)
            y[k, :] = reward_batch[k, :] + np.clip(value, self.value_min, self.value_max)

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
    def train_critic_mc(self, critic, batch_number, epoch, batch_size, validation_ratio, learning_rate):

        # creating buffer
        train_buffer = []
        valid_buffer = []

        index = np.arange(batch_number)
        np.random.shuffle(index)
        slice_var = int(validation_ratio * len(index))
        train_index = index[slice_var:]

        # directory is hard codes... may change later **
        directory = os.getcwd()
        os.chdir(directory + '/Plant data')
        for k in range(batch_number):
            state_data = np.loadtxt("PL_state" + str(k) + ".txt")
            input_data = np.loadtxt("PL_input" + str(k) + ".txt")
            reward_data = np.loadtxt("PL_reward" + str(k) + ".txt")
            scaled_input_data = np.zeros_like(input_data)
            scaled_state_data = np.zeros_like(state_data)
            for kk in range(TIME_STEP + 1):
                scaled_state_data[:, kk] = UtilityFunctions.scale(state_data[:, kk], self.state_min, self.state_max)
                scaled_input_data[:, kk] = UtilityFunctions.scale(input_data[:, kk], self.input_min, self.input_max)
            for kk in range(TIME_STEP):
                data_tuple = [scaled_state_data[:, kk].tolist(), scaled_input_data[:, kk], reward_data[kk],
                              scaled_state_data[:, kk + 1].tolist()]
                if k in train_index:
                    train_buffer.append(data_tuple)
                else:
                    valid_buffer.append(data_tuple)
        os.chdir(directory)

        train_ex = np.zeros((len(train_buffer), self.state_dim + self.input_dim))
        train_la = np.zeros((len(train_buffer), 1))
        for k in range(len(train_buffer)//TIME_STEP):  # For each batch
            for kk in range(TIME_STEP):  # For each time step
                train_ex[TIME_STEP*k + kk, :self.state_dim] = train_buffer[TIME_STEP*k + kk][0]
                train_ex[TIME_STEP*k + kk, self.state_dim:self.state_dim + self.input_dim] \
                    = train_buffer[TIME_STEP*k + kk][1]
                mc_value = 0
                for kkk in range(kk, TIME_STEP):
                    mc_value += train_buffer[TIME_STEP*k + kkk][2]
                mc_value += self._scaled_terminal_cost(train_buffer[TIME_STEP*(k+1) - 1][3], 0)  # arbitrary input
                train_la[TIME_STEP*k + kk, :] = mc_value

        valid_ex = np.zeros((len(valid_buffer), self.state_dim + self.input_dim))
        valid_la = np.zeros((len(valid_buffer), 1))
        for k in range(len(valid_buffer)//TIME_STEP):
            for kk in range(TIME_STEP):
                valid_ex[TIME_STEP*k + kk, :self.state_dim] = valid_buffer[TIME_STEP*k + kk][0]
                valid_ex[TIME_STEP*k + kk, self.state_dim:self.state_dim + self.input_dim] \
                    = valid_buffer[TIME_STEP*k + kk][1]
                mc_value = 0
                for kkk in range(kk, TIME_STEP):
                    mc_value += valid_buffer[TIME_STEP*k + kkk][2]
                mc_value += self._scaled_terminal_cost(valid_buffer[TIME_STEP*(k+1) - 1][3], 0)  # arbitrary input
                valid_la[TIME_STEP*k + kk, :] = mc_value

        # optimization compile  # change later #################
        critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        critic.compile(
            loss='mse',
            optimizer=critic_optimizer,
            metrics=['mae', 'mse']
        )

        history = critic.fit(train_ex, train_la, epochs=epoch, batch_size=batch_size,
                             validation_data=(valid_ex, valid_la))
        mae = np.array(history.history["mae"])
        mse = np.array(history.history["mse"])
        val_mae = np.array(history.history["val_mae"])
        val_mse = np.array(history.history["val_mse"])
        nn_weight = critic.get_weights()

        return nn_weight, mae, mse, val_mae, val_mse, critic

    def _neural_network_casadi(self, x, nn_weight):

        def activation(xx):
            x_len, _ = xx.shape
            for kk in range(x_len):
                xx[kk, :] = ca.log(1 + xx[kk, :] * xx[kk, :])
                y = xx
            return y

        out = x
        for k in range(len(self.node_number)):
            out = activation(ca.transpose(ca.mtimes(ca.transpose(out), nn_weight[2*k])) + nn_weight[2*k+1])
        return out


if __name__ == '__main__':
    a = DNN(1, 1, [4, 4, 2, 1], 100, 0.02)
    z = a.build_critic()
    print(z.summary())


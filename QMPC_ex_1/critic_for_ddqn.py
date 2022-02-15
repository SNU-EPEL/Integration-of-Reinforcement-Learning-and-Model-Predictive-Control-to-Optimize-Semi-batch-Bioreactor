import numpy as np
import tensorflow as tf
import UtilityFunctions

from tensorflow import keras
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import layers


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
        self.value_min = -10.
        self.value_max = 10.
        self.action_bound = 1.0

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

    def q_minimization(self, critic, state, action_number1, action_number2, ref_act):
        # scaled in, scaled out
        # This code is designed for 2-dim action only, please change it later....!!!!!!
        act_can1 = np.linspace(0, 1, num=action_number1)
        act_can2 = np.linspace(0, 1, num=action_number2)
        q_values = np.zeros((action_number1, action_number2))
        for k in range(action_number1):
            for kk in range(action_number2):
                action_now = np.array([act_can1[k], act_can2[kk]])
                if np.linalg.norm(action_now - ref_act)/2 < self.action_bound:
                    scaled_input = np.array([action_now])
                    # print("s_i", scaled_input.shape) # (1, 1)
                    sa_input = np.hstack([state, scaled_input])
                    q_values[k, kk] = critic(sa_input, training=False)
                    #  print('sa_input', sa_input)
                    #  print('Q1', critic(sa_input, training=False))
                else:
                    q_values[k, kk] = self.value_max
        # print('Q', q_values)
        # print(sa_input, q_values)
        min_index = np.unravel_index(np.argmin(q_values, axis=None), q_values.shape)
        neural_network_value = q_values[min_index]
        act = np.array([act_can1[min_index[0]], act_can2[min_index[1]]])
        return act, neural_network_value

    def _scaled_terminal_cost(self, state, action):
        return self.terminal_cost(UtilityFunctions.descale(state, self.state_min, self.state_max),
                                  UtilityFunctions.descale(action, self.input_min, self.input_max))

'''
if __name__ == '__main__':

    def tcost(x):
        return x

    a = DNN(2, 2, [4, 4, 1], 10000, 0.002, tcost)
    c = a.build_critic()
    s = np.zeros((1, 2))
    print(s)
    ra = np.zeros((1, 2))
    print(ra)
    aa, nn = a.q_minimization(c, s, 3, 3, ra)
    print(aa, nn)
'''

# 2021.04.13 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# Utility functions

import numpy as np


def plant_state_to_local_state(plant_state):
    local_state = np.array([plant_state[0], plant_state[1] + plant_state[2] + plant_state[3] + plant_state[4],
                            plant_state[6], plant_state[7], plant_state[8]])
    return local_state


def local_input_to_plant_input(local_input, time_index):
    ratio = np.loadtxt('Feed_ratio.txt')
    plant_input = np.array([local_input / (1 + 5/3*ratio[time_index]),
                            ratio[time_index]*local_input/(1 + 5/3*ratio[time_index])])
    return plant_input


def plant_input_to_local_input(plant_input):
    local_input = plant_input[0] + 5 / 3*plant_input[1]
    return local_input


# scale X -> [0,1]
def scale(var, var_min, var_max):
    scaled_var = (var - var_min) / (var_max - var_min)
    return scaled_var


# descale [0,1] - > X
def descale(var, var_min, var_max):
    descaled_var = (var_max - var_min) * var + var_min
    return descaled_var

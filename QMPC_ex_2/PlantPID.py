# 2021.04.13 written by Tae Hoon Oh, E-mail: rozenk@snu.ac.kr
# PID code


class PID(object):
    def __init__(self, p_gain, i_gain, d_gain, bias):
        self.error_prior = 0
        self.integral_prior = 0
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.bias = bias

    def control(self, error, iteration_time):
        integral = self.integral_prior + error*iteration_time
        derivative = (error - self.error_prior)/iteration_time
        self.error_prior = error
        self.integral_prior = integral
        output = self.p_gain*error + self.i_gain*integral + self.d_gain*derivative + self.bias
        return output

    def reset(self):
        self.error_prior = 0
        self.integral_prior = 0






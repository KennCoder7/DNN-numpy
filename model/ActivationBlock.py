"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np
from utils import *


class Activation(object):
    def __init__(self, name, method):
        self.name = name
        self.__method = method
        self.__input_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim
        return self.name, self.__input_dim

    def forward(self, _x_set):
        _a_set = activation_function(self.__method, _x_set)
        return _a_set

    def backward(self, _e_set, _z_down_set):
        _e_down_set = derivative_function(self.__method, _z_down_set) * _e_set
        return _e_down_set


if __name__ == '__main__':
    x_set = np.random.randn(3, 5)
    print(sigmoid(x_set).shape)
    print(d_sigmoid(x_set).shape)
    print(relu(x_set).shape)
    print(d_relu(x_set).shape)
    print(softmax(x_set).shape)

    # for i in range(3):
    #     print(np.sum(softmax(x_set)[i]))

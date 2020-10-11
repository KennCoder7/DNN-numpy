"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np


class Activation(object):
    def __init__(self, name, method):
        self.name = name
        self.__method = method
        self.__input_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim
        return self.name, self.__input_dim

    def forward(self, x):
        _a = activation_function(self.__method, x)
        return _a

    def backward(self, e, z_down):
        _e_down = derivative_function(self.__method, z_down) * e
        return _e_down


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    shift_x = x - np.max(x)  # 防止输入增大时输出为nan
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def activation_function(method, x):
    if method == 'relu':
        _a = relu(x)
    elif method == 'sigmoid':
        _a = sigmoid(x)
    elif method == 'softmax':
        _a = softmax(x)
    elif method is None:
        _a = x
    else:
        _a = []
        print("No such activation: {}!".format(method))
        exit(1)
    return _a


def derivative_function(method, x):
    if method == 'relu':
        _d = d_relu(x)
    elif method == 'sigmoid':
        _d = d_sigmoid(x)
    elif method is None:
        _d = 1
    else:
        _d = []
        print("No such activation: {}!".format(method))
        exit(1)
    return _d

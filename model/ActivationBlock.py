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

    def forward(self, _x_set):
        _a_set = activation_function(self.__method, _x_set)
        return _a_set

    def backward(self, _e_set, _z_down_set):
        _e_down_set = derivative_function(self.__method, _z_down_set) * _e_set
        return _e_down_set


def sigmoid(_x_set):
    return 1 / (1 + np.exp(-_x_set))


def d_sigmoid(_x_set):
    return (1 - sigmoid(_x_set)) * sigmoid(_x_set)


def softmax(_x_set):
    x_row_max = _x_set.max(axis=-1)
    x_row_max = x_row_max.reshape(list(_x_set.shape)[:-1] + [1])
    _x_set = _x_set - x_row_max
    x_exp = np.exp(_x_set)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(_x_set.shape)[:-1] + [1])
    return x_exp / x_exp_row_sum


def relu(_x_set):
    return np.maximum(0, _x_set)


def d_relu(_x_set):
    _x_set[_x_set <= 0] = 0
    _x_set[_x_set > 0] = 1
    return _x_set


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


if __name__ == '__main__':
    x_set = np.random.randn(3, 5)
    print(sigmoid(x_set).shape)
    print(d_sigmoid(x_set).shape)
    print(relu(x_set).shape)
    print(d_relu(x_set).shape)
    print(softmax(x_set).shape)

    for i in range(3):
        print(np.sum(softmax(x_set)[i]))
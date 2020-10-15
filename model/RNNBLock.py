"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np
from utils import *


class BasicRNN(object):
    """unidirection, static"""
    def __init__(self, name, units, return_last_step):
        self.name = name
        self.__units = units
        self.__return_last_step = return_last_step

        self.__u = None
        self.__w = None
        self.__b = None

        self.__h_set = None
        self.__s_set = None

        self.__input_dim = None
        self.__step = None
        self.__output_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim  # [step, inputs]
        self.__step = self.__input_dim[0]

        std = np.sqrt(1. / self.__units)  # Xavier initialization
        self.__u = np.random.normal(loc=0., scale=std, size=[self.__units, self.__units])
        self.__b = np.random.normal(loc=0., scale=std, size=[self.__units])
        std = np.sqrt(1. / self.__input_dim[1])  # Xavier initialization
        self.__w = np.random.normal(loc=0., scale=std, size=[self.__input_dim[1], self.__units])

        self.__output_dim = [self.__units] if self.__return_last_step else [self.__step, self.__units]

        return self.name, self.__output_dim

    def weight_shape(self):
        return {'u': self.__u.shape, 'w': self.__w.shape, 'b': self.__b.shape}

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.__input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        shape = _x_set.shape  # [nums, step, inputs]
        nums = shape[0]
        # _x_set = _x_set.cppy()
        _x_set = _x_set.transpose([1, 0, 2])  # [step, nums, inputs]
        _h = np.zeros([self.__step + 1, nums, self.__units])  # [step+1, nums, units] & zero initial state
        _s = np.zeros([self.__step, nums, self.__units])
        for t in range(self.__step):
            t_h = t + 1
            _s[t] = np.dot(_h[t_h - 1], self.__u) + np.dot(_x_set[t], self.__w) + self.__b
            _h[t_h] = tanh(_s[t])
        _z = _h[-1, :, :] if self.__return_last_step else _h[1:, :, :].transpose([1, 0, 2])
        self.__h_set = _h.transpose([1, 0, 2])
        self.__s_set = _s.transpose([1, 0, 2])
        return _z

    def backward(self, _e_set):
        _e_set = _e_set.copy()  # [nums, units] or [nums, step, units]
        nums = _e_set.shape[0]
        if len(_e_set.shape) == 2:  # [nums, units]
            _e_set_temp = np.zeros([self.__step, nums, self.__units])  # [step, nums, units]
            _e_set_temp[self.__step - 1] = _e_set
            _e_set = _e_set_temp  # [step, nums, units]
        else:   # [nums, step, units]
            _e_set = _e_set.transpose([1, 0, 2])    # [step, nums, units]
        _h = self.__h_set.transpose([1, 0, 2])  # [step+1, nums, units]
        _e_down_t_set = np.zeros([self.__step, nums, self.__input_dim[1]])  # [step, nums, inputs]
        for t in range(self.__step):
            t_h = t + 1
            _e_k_set = np.zeros([t + 1, nums, self.__units])
            _e_k_set[t] = np.multiply((1 - _h[t_h] ** 2), _e_set[t])
            for k in range(t - 1, -1, -1):
                k_h = k + 1
                _e_k_set[k] = np.multiply((1 - _h[k_h] ** 2), np.dot(_e_k_set[k + 1], self.__u.transpose()))
                _e_down_t_set[k] += np.dot(_e_k_set[k], self.__w.transpose())
        _e_down_set = _e_down_t_set.transpose([1, 0, 2])  # [nums, step, inputs]
        return _e_down_set

    def gradient(self, _z_down_set, _e_set):
        _z_down_set = _z_down_set.copy()  # [nums, step, inputs]
        _e_set = _e_set.copy()  # [nums, units] or [nums, step, units]
        nums = len(_e_set)
        if len(_e_set.shape) == 2:  # [nums, units]
            _e_set_temp = np.zeros([self.__step, nums, self.__units])  # [step, nums, units]
            _e_set_temp[self.__step - 1] = _e_set
            _e_set = _e_set_temp  # [step, nums, units]
        else:   # [nums, step, units]
            _e_set = _e_set.transpose([1, 0, 2])    # [step, nums, units]
        _h = self.__h_set.transpose([1, 0, 2])  # [step+1, nums, units]
        _x = _z_down_set.transpose([1, 0, 2])    # [step, nums, units]
        _du_t = np.zeros([self.__step, nums, self.__units, self.__units])
        _dw_t = np.zeros([self.__step, nums, self.__input_dim[1], self.__units])
        _db_t = np.zeros([self.__step, nums, self.__units])
        for t in range(self.__step):
            t_h = t + 1
            _e_k_set = np.zeros([t + 1, nums, self.__units])
            _e_k_set[t] = np.multiply((1 - _h[t_h] ** 2), _e_set[t])
            for k in range(t-1, -1, -1):
                k_h = k + 1
                _e_k_set[k] = np.multiply((1 - _h[k_h] ** 2), np.dot(_e_k_set[k+1], self.__u.transpose()))
                _du_t[t] += np.matmul(np.expand_dims(_h[k_h - 1], -1), np.expand_dims(_e_k_set[k], -2))
                _dw_t[t] += np.matmul(np.expand_dims(_x[k], -1), np.expand_dims(_e_k_set[k], -2))
                _db_t[t] += _e_k_set[k]
        _du = np.sum(_du_t, axis=(0, 1)) / nums  # [step, nums, units, units] --> [units, units] / nums
        _dw = np.sum(_dw_t, axis=(0, 1)) / nums  # [inputs, units] / nums
        _db = np.sum(_db_t, axis=(0, 1)) / nums  # [units] / nums
        return {'w': _dw, 'u': _du, 'b': _db}

    def gradient_descent(self, _g, test_lr=1.):
        _du = _g['u']
        _dw = _g['w']
        _db = _g['b']
        self.__u -= test_lr * _du
        self.__w -= test_lr * _dw
        self.__b -= test_lr * _db


if __name__ == '__main__':
    def rnn1_func():
        rnn1 = BasicRNN(name='rnn1', units=10, return_last_step=True)
        rnn1.initial(input_dim=[8, 5])
        x = np.random.randn(3, 8, 5)
        y = np.random.randn(3, 10)

        # y_ = rnn1.forward(x)
        # print(y_.shape)
        # e = y_ - y
        # print(e.shape)
        # e_down = rnn1.backward(e)
        # print(e_down.shape)

        for i in range(101):
            y_ = rnn1.forward(x)
            cost = y_ - y
            g = rnn1.gradient(x, cost)
            rnn1.gradient_descent(g, test_lr=0.01)
            if i % 10 == 0:
                print(f"Epoch{i}: Loss={np.sum(cost ** 2) / 3}")


    def rnn2_func():
        rnn1 = BasicRNN(name='rnn1', units=10, return_last_step=False)
        rnn1.initial(input_dim=[8, 5])
        x = np.random.randn(3, 8, 5)
        y = np.random.randn(3, 8, 10)

        # y_ = rnn1.forward(x)
        # print(y_.shape)
        # e = y_ - y
        # print(e.shape)
        # e_down = rnn1.backward(e)
        # print(e_down.shape)

        for i in range(101):
            y_ = rnn1.forward(x)
            cost = y_ - y
            g = rnn1.gradient(x, cost)
            rnn1.gradient_descent(g, test_lr=0.01)
            if i % 10 == 0:
                print(f"Epoch{i}: Loss={np.sum(cost ** 2) / 3}")

    def rnn3_func():
        rnn1 = BasicRNN(name='rnn1', units=10, return_last_step=False)
        rnn1.initial(input_dim=[8, 5])
        rnn2 = BasicRNN(name='rnn2', units=10, return_last_step=True)
        rnn2.initial(input_dim=[8, 10])
        x = np.random.randn(3, 8, 5)
        y = np.random.randn(3, 10)
        for i in range(101):
            y_1 = rnn1.forward(x)
            y_2 = rnn2.forward(y_1)
            cost2 = y_2 - y
            cost1 = rnn2.backward(cost2)
            g2 = rnn2.gradient(y_1, cost2)
            g1 = rnn1.gradient(x, cost1)
            rnn1.gradient_descent(g1, test_lr=0.01)
            rnn2.gradient_descent(g2, test_lr=0.01)
            if i % 10 == 0:
                print(f"Epoch{i}: Loss={np.sum(cost2 ** 2) / 3}")

    # rnn1_func()
    # rnn2_func()
    rnn3_func()


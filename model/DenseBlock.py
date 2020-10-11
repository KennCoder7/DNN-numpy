"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np


class Dense(object):
    def __init__(self, name, units):
        self.name = name
        self.__units = units
        self.__input_dim = None
        self.__w = None
        self.__b = None

    def initial(self, input_dim):
        if len(input_dim) != 1:
            print("{} initial error!".format(self.name))
            exit(1)
        self.__input_dim = input_dim
        std = np.sqrt(2. / (self.__input_dim[0]))  # he normalization
        self.__w = np.random.normal(loc=0., scale=std, size=[self.__units, self.__input_dim[0]])
        self.__b = np.random.normal(loc=0., scale=std, size=[self.__units])
        return self.name, [self.__units]

    def weight_shape(self):
        return self.__w.shape, self.__b.shape

    def forward(self, x):
        if x.shape[0] != self.__input_dim[0]:
            print("{} input set dim error!".format(self.name))
            exit(1)
        _z = np.dot(self.__w, x) + self.__b
        return _z

    def backward(self, _e):
        # _e_down = np.zeros([self.__input_dim[0]])
        _e_down = np.dot(self.__w.transpose(1, 0), _e)
        return _e_down

    def gradient(self, z_down, _e):
        _e = _e.copy()
        _dw = np.zeros([self.__units, self.__input_dim[0]])
        _db = np.zeros([self.__units])
        _dw = np.outer(_e, z_down)
        _db = _e
        return _dw, _db

    def gradient_descent(self, _dw, _db):
        self.__w -= _dw
        self.__b -= _db


if __name__ == '__main__':
    dense_block = Dense(name='fc', units=10)
    dense_block.initial([5])

    x = np.random.randn(5)
    y = np.random.randn(10)

    for i in range(100):
        y_ = dense_block.forward(x)
        cost = y_ - y
        dw, db = dense_block.gradient(x, cost)
        dense_block.gradient_descent(0.01 * dw, 0.01 * db)
        print(f"Epoch{i}: Loss={np.sum(cost**2)}")

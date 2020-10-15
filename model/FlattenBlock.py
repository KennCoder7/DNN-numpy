"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np


class Flatten(object):
    def __init__(self, name):
        self.name = name
        self.__input_dim = None
        self.__output_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim
        try:
            self.__output_dim = 1
            for _d in self.__input_dim:
                self.__output_dim *= _d
            self.__output_dim = [self.__output_dim]
        except:
            print("{} initial error!".format(self.name))
            exit(1)
        return self.name, self.__output_dim

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.__input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        nums = len(_x_set)
        return _x_set.reshape(nums, -1)

    def backward(self, _e_up_set):
        _e_up_set = np.array(_e_up_set)
        d_lst = [-1]
        for _d in self.__input_dim:
            d_lst.append(_d)
        return np.reshape(_e_up_set, d_lst)


if __name__ == '__main__':
    flatten = Flatten(name='flatten')
    flatten.initial([32, 2, 2])
    x = np.random.randn(10, 32, 2, 2)
    y = flatten.forward(x)
    print(y.shape)
    e = flatten.backward(y)
    print(e.shape)
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
            self.__output_dim = [input_dim[0]*input_dim[1]*input_dim[2]]
        except:
            print("{} initial error!".format(self.name))
            exit(1)
        return self.name, self.__output_dim

    def forward(self, x):
        if x.shape[0] != self.__input_dim[0] or x.shape[1] != self.__input_dim[1] or x.shape[2] != self.__input_dim[2]:
            print("{} input set dim error!".format(self.name))
            exit(1)
        return x.reshape(-1)

    def backward(self, e_up):
        e_up = np.array(e_up)
        return np.reshape(e_up, self.__input_dim)


if __name__ == '__main__':
    flatten = Flatten(name='flatten')
    flatten.initial([32, 2, 2])
    x = np.random.randn(32, 2, 2)
    y = flatten.forward(x)
    print(y.shape)
    e = flatten.backward(y)
    print(e.shape)
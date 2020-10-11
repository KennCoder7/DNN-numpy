"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np


class MaxPooling2D(object):
    """
    pooling_size==pooling_stride
    """

    def __init__(self, name, pooling_size):
        self.name = name
        self.__pooling_size = pooling_size if isinstance(pooling_size, list) else [pooling_size, pooling_size]
        self.__output_dim = None
        self.__input_dim = None

    def initial(self, input_dim):
        self.__input_dim = input_dim
        try:
            self.__output_dim = [input_dim[0],
                                 int(np.ceil(input_dim[1] / self.__pooling_size[0])),
                                 int(np.ceil(input_dim[2] / self.__pooling_size[1]))]
        except:
            print("{} initial error!".format(self.name))
            exit(1)
        return self.name, self.__output_dim

    def forward(self, x):
        if x.shape[0] != self.__input_dim[0] or x.shape[1] != self.__input_dim[1] or x.shape[2] != self.__input_dim[2]:
            print("{} input set dim error!".format(self.name))
            exit(1)
        pool_value = np.zeros(self.__output_dim)
        pool_index = np.zeros(self.__output_dim)
        for ch in range(self.__output_dim[0]):
            for column in range(self.__output_dim[1]):
                for row in range(self.__output_dim[2]):
                    if column != self.__output_dim[1] - 1 and row != self.__output_dim[2] - 1:
                        part_x = x[ch, column * self.__pooling_size[0]:(column + 1) * self.__pooling_size[0],
                                 row * self.__pooling_size[1]:(row + 1) * self.__pooling_size[1]]
                    elif column == self.__output_dim[1] - 1 and row != self.__output_dim[2] - 1:
                        part_x = x[ch, column * self.__pooling_size[0]:,
                                 row * self.__pooling_size[1]:(row + 1) * self.__pooling_size[1]]
                    elif column != self.__output_dim[1] - 1 and row == self.__output_dim[2] - 1:
                        part_x = x[ch, column * self.__pooling_size[0]:(column + 1) * self.__pooling_size[0],
                                 row * self.__pooling_size[1]:]
                    else:
                        part_x = x[ch, column * self.__pooling_size[0]:, row * self.__pooling_size[1]:]
                    pool_value[ch][column][row] = np.max(part_x)
                    pool_index[ch][column][row] = np.argmax(part_x)
        return pool_value, pool_index

    def backward(self, e, pool_index):
        _e_down = np.zeros(self.__input_dim)
        for ch in range(self.__output_dim[0]):  # filters/channel
            for column in range(self.__output_dim[1]):  # width
                for row in range(self.__output_dim[2]):  # height
                    if column != self.__output_dim[1] - 1 and row != self.__output_dim[2] - 1:
                        width_part = self.__pooling_size[0]
                    else:
                        width_part = self.__input_dim[1] - column * self.__pooling_size[0]
                    column_arg = int(pool_index[ch][column][row] % width_part)
                    row_arg = int(pool_index[ch][column][row] // width_part)
                    _e_down[ch,
                       column * self.__pooling_size[0] + column_arg,
                       row * self.__pooling_size[1] + row_arg] = e[ch][column][row]

        return _e_down


if __name__ == '__main__':
    pool = MaxPooling2D(name='p', pooling_size=2)
    _, out_dim = pool.initial([1, 5, 5])
    print(out_dim)
    x = np.random.rand(1, 5, 5)
    y, y_index = pool.forward(x)
    print(y_index)
    print(x, '\n', y,)
    x_bk = pool.backward(y, y_index)
    print(x, '\n', y, '\n', x_bk)

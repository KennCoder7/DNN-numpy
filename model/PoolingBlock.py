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
            self.__output_dim = [int(np.ceil(input_dim[0] / self.__pooling_size[0])),
                                 int(np.ceil(input_dim[1] / self.__pooling_size[1])),
                                 input_dim[2]]
        except:
            print("{} initial error!".format(self.name))
            exit(1)
        return self.name, self.__output_dim

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.__input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        _x_set = _x_set.transpose([0, 3, 1, 2])
        nums = len(_x_set)
        _dim = [nums, self.__output_dim[2], self.__output_dim[0], self.__output_dim[1]]
        pool_value = np.zeros(_dim)
        pool_index = np.zeros(_dim)
        for n in range(nums):
            for ch in range(_dim[1]):
                for r in range(_dim[2]):
                    for c in range(_dim[3]):
                        if r != _dim[2] - 1 and c != _dim[3] - 1:
                            part_x = _x_set[n, ch,
                                     r * self.__pooling_size[0]:(r + 1) * self.__pooling_size[0],
                                     c * self.__pooling_size[1]:(c + 1) * self.__pooling_size[1]]
                        elif r == _dim[2] - 1 and c != _dim[3] - 1:
                            part_x = _x_set[n, ch,
                                     r * self.__pooling_size[0]:,
                                     c * self.__pooling_size[1]:(c + 1) * self.__pooling_size[1]]
                        elif r != _dim[2] - 1 and c == _dim[3] - 1:
                            part_x = _x_set[n, ch,
                                     r * self.__pooling_size[0]:(r + 1) * self.__pooling_size[0],
                                     c * self.__pooling_size[1]:]
                        else:
                            part_x = _x_set[n, ch, r * self.__pooling_size[0]:, c * self.__pooling_size[1]:]
                        pool_value[n][ch][r][c] = np.max(part_x)
                        pool_index[n][ch][r][c] = np.argmax(part_x)
        return pool_value.transpose([0, 2, 3, 1]), pool_index.transpose([0, 2, 3, 1])

    def backward(self, _e_set, pool_index):
        nums = len(_e_set)
        _e_set = _e_set.transpose([0, 3, 1, 2])
        pool_index = pool_index.transpose([0, 3, 1, 2])
        _dim = [nums, self.__input_dim[2], self.__input_dim[0], self.__input_dim[1]]
        _e_down = np.zeros(_dim)
        for n in range(nums):
            for ch in range(self.__output_dim[2]):  # filters/channel
                for r in range(self.__output_dim[0]):  # rows
                    for c in range(self.__output_dim[1]):  # columns
                        if c != self.__output_dim[1] - 1:
                            width_part = self.__pooling_size[0]
                        else:
                            width_part = self.__input_dim[1] - c * self.__pooling_size[0]
                        row_arg = int(pool_index[n][ch][r][c] // width_part)
                        column_arg = int(pool_index[n][ch][r][c] % width_part)
                        _e_down[n, ch,
                                r * self.__pooling_size[0] + row_arg,
                                c * self.__pooling_size[1] + column_arg] \
                            = _e_set[n][ch][r][c]
        return _e_down.transpose([0, 2, 3, 1])


if __name__ == '__main__':
    pool = MaxPooling2D(name='p', pooling_size=2)
    _, out_dim = pool.initial([4, 5, 3])
    # print(out_dim)
    x = np.random.randn(10, 4, 5, 3)
    y, y_index = pool.forward(x)
    print(y.shape)
    # print(x[0][0], '\n', y[0][0], '\n', y_index[0][0])
    x_bk = pool.backward(y, y_index)
    print(x[0, :, :, 0], '\n', y[0, :, :, 0], '\n', x_bk[0, :, :, 0], '\n', y_index[0, :, :, 0])

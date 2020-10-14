"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np


class Conv2D(object):
    """
    input_dim=[Channel, Width, Height]
    kernel_size=[kw, kh]
    stride=[sw, sh]
    filters=numbers of kernel
    padding='valid', 'same'
    ATTENTION! Current Implementation:
    stride = 1  # for temporary implementation
    """

    def __init__(self, name, kernel_size, filters, padding='valid'):
        self.name = name
        self.__kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size, kernel_size]
        # stride = 1  # for temporary implementation
        # self.__stride = stride if isinstance(stride, list) else [stride, stride]
        self.__filters = filters
        self.__padding = padding
        self.__output_dim = None
        self.__w = None
        self.__b = None
        self.__in_dim = None

    def __get_input_dim(self, input_dim):
        if len(input_dim) != 3:
            print("{} initial error!".format(self.name))
            exit(1)
        self.__input_dim = input_dim

    def __compute_output_dim(self):
        if self.__padding == 'valid':
            ow = self.__input_dim[1] - self.__kernel_size[0] + 1
            oh = self.__input_dim[2] - self.__kernel_size[1] + 1
        elif self.__padding == 'same':
            ow = self.__input_dim[1]
            oh = self.__input_dim[2]
        else:
            ow = 0
            oh = 0
            print('No such padding method :{}!'.format(self.__padding))
            exit(1)
        self.__output_dim = [self.__filters, ow, oh]

    def __initial_weights(self):
        std = np.sqrt(2. /
                      (self.__input_dim[0] * self.__kernel_size[0] * self.__kernel_size[1]))  # he normalization
        self.__w = np.random.normal(loc=0., scale=std,
                                    size=[self.__output_dim[0], self.__input_dim[0],
                                          self.__kernel_size[0], self.__kernel_size[1]])
        self.__b = np.random.normal(loc=0., scale=std, size=[self.__filters])

    def __padding_forward(self, _x_set):
        if self.__kernel_size[0] % 2 == 0:
            left_padding = int(self.__kernel_size[0] / 2 - 1)
            right_padding = int(self.__kernel_size[0] / 2 + 1)
        else:
            left_padding = int(self.__kernel_size[0] // 2)
            right_padding = int(self.__kernel_size[0] // 2)
        if self.__kernel_size[1] % 2 == 0:
            top_padding = int(self.__kernel_size[1] / 2 - 1)
            bottom_padding = int(self.__kernel_size[1] / 2 + 1)
        else:
            top_padding = int(self.__kernel_size[1] // 2)
            bottom_padding = int(self.__kernel_size[1] // 2)
        x_padding = np.zeros([len(_x_set),
                              self.__input_dim[0],
                              self.__input_dim[1] + left_padding + right_padding,
                              self.__input_dim[2] + top_padding + bottom_padding])
        x_padding[:, :, left_padding:self.__input_dim[1] + left_padding,
        top_padding:self.__input_dim[2] + top_padding] = _x_set.copy()
        return x_padding

    def __padding_backward(self, _e_set):
        if self.__padding == 'same':
            if self.__kernel_size[0] % 2 == 0:
                left_padding = int(self.__kernel_size[0] / 2 - 1)
                right_padding = int(self.__kernel_size[0] / 2 + 1)
            else:
                left_padding = int(self.__kernel_size[0] // 2)
                right_padding = int(self.__kernel_size[0] // 2)
            if self.__kernel_size[1] % 2 == 0:
                top_padding = int(self.__kernel_size[1] / 2 - 1)
                bottom_padding = int(self.__kernel_size[1] / 2 + 1)
            else:
                top_padding = int(self.__kernel_size[1] // 2)
                bottom_padding = int(self.__kernel_size[1] // 2)
            e_padding = np.zeros([len(_e_set), self.__output_dim[0],
                                  self.__output_dim[1] + left_padding + right_padding,
                                  self.__output_dim[2] + top_padding + bottom_padding])
            e_padding[:, :, left_padding:self.__output_dim[1] + left_padding,
            top_padding:self.__output_dim[2] + top_padding] = _e_set.copy()
        else:
            left_padding = int(self.__kernel_size[0] - 1)
            right_padding = int(self.__kernel_size[0] - 1)
            top_padding = int(self.__kernel_size[1] - 1)
            bottom_padding = int(self.__kernel_size[1] - 1)
            e_padding = np.zeros([len(_e_set), self.__output_dim[0],
                                  self.__output_dim[1] + left_padding + right_padding,
                                  self.__output_dim[2] + top_padding + bottom_padding])
            e_padding[:, :, left_padding:self.__output_dim[1] + left_padding,
            top_padding:self.__output_dim[2] + top_padding] = _e_set.copy()
        return e_padding

    @staticmethod
    def __dw_conv(_z_down_set, _e_set):
        ks = [_e_set.shape[0], _e_set.shape[1]]
        _z_down_set = _z_down_set.copy()
        nums = _z_down_set.shape[2]
        rows = _z_down_set.shape[0] - ks[0] + 1
        columns = _z_down_set.shape[1] - ks[1] + 1
        _z = np.zeros([rows, columns, nums])
        for r in range(rows):  # rows of output
            for c in range(columns):  # columns of output
                part_x = _z_down_set[r:r + ks[0], c:c + ks[1], :]
                _z[r][c] = np.sum(np.multiply(part_x, _e_set), axis=(0, 1))
        return _z

    @staticmethod
    def __matrix_conv(_x_set, _kernel):
        if _x_set.shape[1] != _kernel.shape[1]:
            print("matrix_conv error!")
            exit(1)
        ks = [_kernel.shape[2], _kernel.shape[3]]
        _x_set = _x_set.copy()
        _kernel = _kernel.copy()
        nums = len(_x_set)
        filters = _kernel.shape[0]
        rows = _x_set.shape[2] - ks[0] + 1
        columns = _x_set.shape[3] - ks[1] + 1
        _z = np.zeros([rows, columns, nums, filters])
        _kernel = _kernel.transpose([1, 2, 3, 0])
        for r in range(rows):  # rows of output
            for c in range(columns):  # columns of output
                part_x = _x_set[:, :, r:r + ks[0], c:c + ks[1]]
                _z[r][c] = np.dot(part_x.reshape(nums, -1), _kernel.reshape(-1, filters))
        return _z.transpose([2, 3, 0, 1])

    def __w_flip180(self):
        arr = self.__w.copy()
        new_arr = arr.reshape(-1)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        new_arr = new_arr.transpose(1, 0, 2, 3)
        return new_arr[:, ::-1, :, :]

    def initial(self, input_dim):
        self.__get_input_dim(input_dim)
        self.__compute_output_dim()
        self.__initial_weights()
        return self.name, self.__output_dim

    def weight_shape(self):
        return {'w': self.__w.shape, 'b': self.__b.shape}

    def forward(self, _x_set):
        if list(_x_set.shape[1:]) != list(self.__input_dim):
            print("{} input set dim error!".format(self.name))
            exit(1)
        _x_set = _x_set.copy() if self.__padding == 'valid' else self.__padding_forward(_x_set)
        _z = self.__matrix_conv(_x_set, self.__w)
        return _z

    def backward(self, _e_set):
        _e_set = self.__padding_backward(_e_set)
        _w_flp = self.__w_flip180()
        # print(self.w[-1, 0], '\n', _w_flp[0, -1])
        _e_down_set = self.__matrix_conv(_e_set, _w_flp)
        return _e_down_set

    def gradient(self, _z_down_set, _e_set):
        _e_set = _e_set.copy()
        _z_down_set = _z_down_set.copy() if self.__padding == 'valid' else self.__padding_forward(_z_down_set.copy())
        nums = len(_z_down_set)
        _w_shape = list(self.__w.shape)
        _w_shape.append(nums)
        _dw = np.zeros(_w_shape)
        _e_set = _e_set.transpose([1, 2, 3, 0])
        _z_down_set = _z_down_set.transpose([1, 2, 3, 0])
        for m in range(self.__w.shape[0]):
            for n in range(self.__w.shape[1]):
                _dw[m][n] = self.__dw_conv(_z_down_set[n], _e_set[m])
        _dw = np.sum(_dw, axis=-1) / nums
        _db = np.sum(_e_set, (1, 2, 3)) / nums
        return {'w': _dw, 'b': _db}

    def gradient_descent(self, _g, test_lr=1.):
        _dw = _g['w']
        _db = _g['b']
        self.__w -= test_lr * _dw
        self.__b -= test_lr * _db


if __name__ == '__main__':
    cnn_block = Conv2D(name='c', kernel_size=3, filters=8, padding='same')
    cnn_block.initial([2, 5, 5])

    x = np.random.randn(1, 2, 5, 5)
    y = np.random.randn(1, 8, 5, 5)

    for i in range(100):
        y_ = cnn_block.forward(x)
        cost = y_ - y
        # e = cnn_block.backward(cost)
        g = cnn_block.gradient(x, cost)
        cnn_block.gradient_descent(g, 0.01)
        print(f"Epoch{i}: Loss={np.sum(cost**2)/len(x)}")

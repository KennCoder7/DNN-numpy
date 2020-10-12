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

    def __padding_forward(self, _x):
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
        x_padding = np.zeros([self.__input_dim[0],
                              self.__input_dim[1] + left_padding + right_padding,
                              self.__input_dim[2] + top_padding + bottom_padding])
        x_padding[:, left_padding:self.__input_dim[1] + left_padding,
        top_padding:self.__input_dim[2] + top_padding] = _x.copy()
        return x_padding

    def __padding_backward(self, _x):
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
            x_padding = np.zeros([self.__output_dim[0],
                                  self.__output_dim[1] + left_padding + right_padding,
                                  self.__output_dim[2] + top_padding + bottom_padding])
            x_padding[:, left_padding:self.__output_dim[1] + left_padding,
            top_padding:self.__output_dim[2] + top_padding] = _x.copy()
        else:
            left_padding = int(self.__kernel_size[0] - 1)
            right_padding = int(self.__kernel_size[0] - 1)
            top_padding = int(self.__kernel_size[1] - 1)
            bottom_padding = int(self.__kernel_size[1] - 1)
            x_padding = np.zeros([self.__output_dim[0],
                                  self.__output_dim[1] + left_padding + right_padding,
                                  self.__output_dim[2] + top_padding + bottom_padding])
            x_padding[:, left_padding:self.__output_dim[1] + left_padding,
            top_padding:self.__output_dim[2] + top_padding] = _x.copy()
        return x_padding

    @staticmethod
    def __conv(_x, _kernel):
        ks = [_kernel.shape[0], _kernel.shape[1]]
        _x = _x.copy()
        width = _x.shape[0]-ks[0]+1
        height = _x.shape[1]-ks[1]+1
        _z = np.zeros([width, height])
        for column in range(width):  # width of output
            for row in range(height):  # height of output
                part_x = _x[column:column + ks[0], row:row + ks[1]]
                _z[column][row] = np.vdot(_kernel, part_x)
        return _z

    @staticmethod
    def __matrix_conv(_x, _kernel):
        if _x.shape[0] != _kernel.shape[1]:
            print("matrix_conv error!")
            exit(1)
        ks = [_kernel.shape[2], _kernel.shape[3]]
        _x = _x.copy()
        filters = _kernel.shape[0]
        width = _x.shape[1] - ks[0] + 1
        height = _x.shape[2] - ks[1] + 1
        _z = np.zeros([filters, width, height])
        for ch in range(filters):  # filters
            for column in range(width):  # width of output
                for row in range(height):  # height of output
                    part_x = _x[:, column:column + ks[0], row:row + ks[1]]
                    _z[ch][column][row] = np.vdot(_kernel[ch], part_x)
        return _z

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
        return self.__w.shape, self.__b.shape

    def forward(self, _x):
        if _x.shape[0] != self.__input_dim[0] or _x.shape[1] != self.__input_dim[1] or \
                _x.shape[2] != self.__input_dim[2]:
            print("{} input set dim error!".format(self.name))
            exit(1)
        _x = _x.copy() if self.__padding == 'valid' else self.__padding_forward(_x)
        _z = self.__matrix_conv(_x, self.__w)
        return _z

    def backward(self, _e):
        _e = self.__padding_backward(_e)
        _w_flp = self.__w_flip180()
        # print(self.w[-1, 0], '\n', _w_flp[0, -1])
        _e_down = self.__matrix_conv(_e, _w_flp)
        return _e_down

    def gradient(self, z_down, _e):
        _e = _e.copy()
        _z_down = z_down.copy() if self.__padding == 'valid' else self.__padding_forward(z_down.copy())
        _ch = z_down.shape[0]
        _dw = np.zeros(self.__w.shape)
        for m in range(self.__w.shape[0]):
            for n in range(self.__w.shape[1]):
                _dw[m][n] = self.__conv(_z_down[n], _e[m])
        _db = np.sum(_e, (1, 2))
        return _dw, _db

    def gradient_descent(self, _dw, _db):
        self.__w -= _dw
        self.__b -= _db


if __name__ == '__main__':
    cnn_block = Conv2D(name='c', kernel_size=3, filters=8, padding='valid')
    cnn_block.initial([2, 5, 5])

    x = np.random.randn(2, 5, 5)
    y = np.random.randn(8, 3, 3)
    for i in range(100):
        y_ = cnn_block.forward(x)
        cost = y_ - y
        # e = cnn_block.backward(cost)
        dw, db = cnn_block.gradient(x, cost)
        cnn_block.gradient_descent(0.01*dw, 0.01*db)
        print(f"Epoch{i}: Loss={np.sum(cost**2)}")

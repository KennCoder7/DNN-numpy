"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
import numpy as np
from ConvBlock import Conv2D
from PoolingBlock import MaxPooling2D
from FlattenBlock import Flatten
from DenseBlock import Dense
from ActivationBlock import Activation


class Model(object):
    def __init__(self, name, input_dim, n_class=None):
        self.name = name
        self.__n_class = n_class
        self.__input_dim = input_dim
        self.__dw = {}
        self.__db = {}
        self.__z = {}
        self.__e = {}
        self.__pool_index = {}
        self.__layer_block_dct = {}
        self.__layer_name_lst = ['x']
        self.__layer_output_dim_lst = [input_dim]

        self.__train_x_set = None
        self.__train_y_set = None
        # self.__test_x_set = None
        # self.__test_y_set = None

        self.__train_loss_log = []
        self.__train_acc_log = []

    def initial(self, block):
        temp_dim = self.__input_dim
        for i, layer_block in enumerate(block):
            name, temp_dim = layer_block.initial(temp_dim)
            if name not in self.__layer_name_lst:
                self.__layer_name_lst.append(name)
                self.__layer_output_dim_lst.append(temp_dim)
                self.__layer_block_dct[name] = layer_block
            else:
                print('Repeated Layer Name: {}!'.format(name))
                exit(1)
        self.print_structure()

    def print_structure(self):
        for i in range(len(self.__layer_name_lst)):
            print("{}:Layer[{}] Output Dim={}".format(self.name,
                                                      self.__layer_name_lst[i], self.__layer_output_dim_lst[i]))

    def __forward(self, _x_set):
        temp_z_set = _x_set.copy()
        self.__z['x'] = temp_z_set
        for layer_block in self.__layer_block_dct.values():
            if isinstance(layer_block, (Conv2D, Dense, Flatten, Activation)):
                temp_z_set = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set
            elif isinstance(layer_block, MaxPooling2D):
                temp_z_set, self.__pool_index[layer_block.name] = layer_block.forward(temp_z_set)
                self.__z[layer_block.name] = temp_z_set

    def __backward(self, _target_set):
        _y_set = self.__z[self.__layer_name_lst[-1]]
        self.__e[self.__layer_name_lst[-1]] = np.sum(-_target_set * np.log(_y_set + 1e-8))
        self.__e[self.__layer_name_lst[-2]] = self.__cross_entropy_cost(_y_set, _target_set)
        for i in range(len(self.__layer_name_lst) - 2, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense, Flatten)):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set)
            elif isinstance(layer_block, MaxPooling2D):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__pool_index[layer_name])
            elif isinstance(layer_block, Activation):
                _e_set = self.__e[layer_name]
                self.__e[layer_name_down] = layer_block.backward(_e_set, self.__z[layer_name_down])

    @staticmethod
    def __cross_entropy_cost(_y_set, _target_set):
        prd_prb = _y_set.copy()
        if len(prd_prb) != len(_target_set):
            print("Cross entropy error!")
            exit(1)
        return prd_prb - _target_set

    def __gradient(self, _x_set, _target_set):
        _dw = {}
        _db = {}
        self.__forward(_x_set)
        self.__backward(_target_set)
        _batch_train_loss = self.__loss_of_current() / len(_x_set)
        _batch_train_acc = 0
        for i in range(len(_x_set)):
            if np.argmax(self.__z[self.__layer_name_lst[-1]][i]) == np.argmax(_target_set[i]):
                _batch_train_acc += 1
        _batch_train_acc /= len(_x_set)
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_name_down = self.__layer_name_lst[i - 1]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense)):
                _z_down = self.__z[layer_name_down]
                _e = self.__e[layer_name]
                _dw[layer_name], _db[layer_name] = layer_block.gradient(_z_down, _e)
        return _dw, _db, _batch_train_loss, _batch_train_acc

    def __gradient_descent(self, _dw, _db):
        for i in range(len(self.__layer_name_lst) - 1, 0, -1):
            layer_name = self.__layer_name_lst[i]
            layer_block = self.__layer_block_dct[layer_name]
            if isinstance(layer_block, (Conv2D, Dense)):
                layer_block.gradient_descent(_dw[layer_name], _db[layer_name])

    def fit(self, train_x_set, train_y_set):
        self.__train_x_set = train_x_set
        self.__train_y_set = train_y_set

    @staticmethod
    def __shuffle_set(sample_set, target_set):
        index = np.arange(len(sample_set))
        np.random.shuffle(index)
        return sample_set[index], target_set[index]

    def train(self, lr, momentum=0.9, max_epoch=1000, batch_size=64, shuffle=True, interval=100):
        """
        Training model by SGD optimizer.
        :param lr: learning rate
        :param momentum: momentum rate
        :param max_epoch: max epoch
        :param batch_size: batch size
        :param shuffle: whether shuffle training set
        :param interval: print training log each interval
        :return: none
        """
        if self.__train_x_set is None:
            print("None data fit!")
            exit(1)
        vw = {}
        vb = {}
        for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
            if isinstance(layer_block, (Conv2D, Dense)):
                w_shape, b_shape = layer_block.weight_shape()
                vw[layer_name] = np.zeros(w_shape)
                vb[layer_name] = np.zeros(b_shape)
        batch_nums = len(self.__train_x_set) // batch_size
        for e in range(max_epoch):
            if shuffle and e % batch_nums == 0:
                self.__shuffle_set(self.__train_x_set, self.__train_y_set)
            start_index = e % batch_nums * batch_size
            t_x = self.__train_x_set[start_index:start_index + batch_size]
            t_y = self.__train_y_set[start_index:start_index + batch_size]
            _dw, _db, _batch_train_loss, _batch_train_acc = self.__gradient(t_x, t_y)
            for layer_name, layer_block in zip(self.__layer_block_dct.keys(), self.__layer_block_dct.values()):
                if isinstance(layer_block, (Conv2D, Dense)):
                    vw[layer_name] = momentum * vw[layer_name] - lr * _dw[layer_name]
                    vb[layer_name] = momentum * vb[layer_name] - lr * _db[layer_name]
                    _dw[layer_name] = -vw[layer_name]
                    _db[layer_name] = -vb[layer_name]
            self.__gradient_descent(_dw, _db)
            if interval and e % interval == 0:
                # print the training log of whole training set rather than batch:
                # train_acc = self.measure(self.__train_x_set, self.__train_y_set)
                self.__train_loss_log.append(_batch_train_loss)
                self.__train_acc_log.append(_batch_train_acc)
                print('Epoch[{}] Batch[{}] Batch_Train_Loss=[{}] Batch_Train_Acc=[{}]'
                      .format(e, e % batch_nums, _batch_train_loss, _batch_train_acc))

    def predict(self, _x_set):
        self.__forward(_x_set)
        return np.argmax(self.__z[self.__layer_name_lst[-1]], axis=-1)

    def __loss_of_current(self):
        return self.__e[self.__layer_name_lst[-1]]

    def measure(self, _x_set, _target_set):
        _prd_set = self.predict(_x_set)
        _target_set = np.argmax(_target_set, axis=-1)
        _acc = 0
        for i in range(len(_x_set)):
            if _prd_set[i] == _target_set[i]:
                _acc += 1
        return _acc / len(_x_set)


if __name__ == '__main__':
    def model_test2():
        _model = Model(name='TEST', input_dim=[3, 10, 10])
        _model.initial(
            [
                Conv2D(name='C1', kernel_size=[3, 3], filters=16, padding='valid'),
                Activation(name='A1', method='relu'),
                MaxPooling2D(name='P1', pooling_size=[2, 2]),
                Conv2D(name='C2', kernel_size=[3, 3], filters=32, padding='valid'),
                Activation(name='A2', method='relu'),
                Flatten(name='flatten'),
                Dense(name='fc1', units=100),
                Activation(name='A3', method='relu'),
                Dense(name='fc2', units=10),
                Activation(name='A4', method='softmax'),
            ]
        )
        _model.print_structure()
        x_set = np.random.randn(2, 3, 10, 10)
        y_set = np.zeros([2, 10])
        y_set[0, 2] = 1
        y_set[1, 7] = 1

        _model.fit(x_set, y_set)
        _model.train(lr=0.01, batch_size=1, max_epoch=100, interval=1)
        print(_model.predict(x_set))


    model_test2()

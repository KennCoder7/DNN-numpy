import numpy as np
import struct
import os

PATH = r'D:\pyproject\ML\datasets\mnist_data'


class MNIST(object):
    def __init__(self, shuffle=True, dimension=3):
        self.train_x_set = None
        self.train_y_set = None
        self.train_labels_set = None

        self.test_x_set = None
        self.test_y_set = None
        self.test_labels_set = None

        self._shuffle = shuffle
        self.__dim = dimension

        self.__load_mnist_train(PATH)
        self.__load_mnist_test(PATH)
        self.__one_hot()
        self.__dimension()
        if self._shuffle:
            self.shuffle()
        self.__normalization()

    def __load_mnist_train(self, path, kind='train'):
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.train_x_set = images
        self.train_labels_set = labels

    def __load_mnist_test(self, path, kind='t10k'):
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.test_x_set = images
        self.test_labels_set = labels

    def __one_hot(self):
        trn = np.zeros([len(self.train_labels_set), 10])
        te = np.zeros([len(self.test_labels_set), 10])
        for i, x in enumerate(self.train_labels_set):
            trn[i, x] = 1
        for i, x in enumerate(self.test_labels_set):
            te[i, x] = 1
        self.train_y_set = trn
        self.test_y_set = te

    def __normalization(self):
        self.train_x_set = self.train_x_set / 255.
        self.test_x_set = self.test_x_set / 255.
        if self.__dim == 3:
            mean = 0
            std = 0
            for x in self.train_x_set:
                mean += np.mean(x[:, :, 0])
            mean /= len(self.train_x_set)
            self.train_x_set -= mean
            for x in self.train_x_set:
                std += np.mean(np.square(x[:, :, 0]).flatten())
            std = np.sqrt(std / len(self.train_x_set))
            print('The mean and std of MNIST:', mean, std)    # 0.1306604762738434 0.30810780385646314
            self.train_x_set /= std
            self.test_x_set -= mean
            self.test_x_set /= std

    def __dimension(self):
        if self.__dim == 1:
            pass
        elif self.__dim == 3:
            self.train_x_set = np.reshape(self.train_x_set, [len(self.train_x_set), 28, 28, 1])
            self.test_x_set = np.reshape(self.test_x_set, [len(self.test_x_set), 28, 28, 1])
        else:
            print('Dimension Error!')
            exit(1)

    def shuffle(self):
        index = np.arange(len(self.train_x_set))
        np.random.seed(7)
        np.random.shuffle(index)
        self.train_x_set = self.train_x_set[index]
        self.train_y_set = self.train_y_set[index]
        self.train_labels_set = self.train_labels_set[index]

    def dimension(self):
        return self.train_x_set.shape[1:]

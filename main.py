"""
author: Kun Wang (Kenn)
e-mail: iskenn7@gmail.com
"""
from dataset.MNIST import MNIST
from model.Model import *


def mlp_mnist():
    mnist = MNIST(dimension=3)
    model = Model(name='model', input_dim=[28, 28, 1])
    model.initial(
        [
            Flatten(name='flatten'),
            Dense(name='fc1', units=100),
            Activation(name='A1', method='relu'),
            Dense(name='fc2', units=100),
            Activation(name='A2', method='relu'),
            Dense(name='fc3', units=10),
            Activation(name='A3', method='softmax'),
        ]
    )
    model.fit(mnist.train_x_set, mnist.train_y_set)
    model.train(lr=0.1, momentum=0.9, max_epoch=500, batch_size=128, interval=10)
    print('Test_Acc=[{}]'.format(model.measure(mnist.test_x_set, mnist.test_y_set)))


def cnn_mnist():
    mnist = MNIST(dimension=3)
    model = Model(name='model', input_dim=[28, 28, 1])
    model.initial(
        [
            Conv2D(name='C1', kernel_size=[3, 3], filters=5, padding='valid'),
            Activation(name='A1', method='relu'),
            MaxPooling2D(name='P1', pooling_size=[2, 2]),
            Flatten(name='flatten'),
            Dense(name='fc1', units=100),
            Activation(name='A3', method='relu'),
            Dense(name='fc2', units=10),
            Activation(name='A4', method='softmax'),
        ]
    )
    model.fit(mnist.train_x_set, mnist.train_y_set)
    model.train(lr=0.01, momentum=0.9, max_epoch=500, batch_size=128, interval=10)
    print('Test_Acc=[{}]'.format(model.measure(mnist.test_x_set, mnist.test_y_set)))


def rnn_mnist():
    mnist = MNIST(dimension=3)
    model = Model(name='model', input_dim=[28, 28])
    model.initial(
        [
            BasicRNN(name='R1', units=10, return_last_step=False),
            Activation(name='A1', method='relu'),
            BasicRNN(name='R2', units=10, return_last_step=False),
            Activation(name='A2', method='relu'),
            Flatten(name='flatten'),
            Dense(name='fc1', units=10),
            Activation(name='A3', method='softmax'),
        ]
    )
    model.fit(mnist.train_x_set.squeeze(-1), mnist.train_y_set)
    model.train(lr=0.01, momentum=0.9, max_epoch=500, batch_size=128, interval=10)
    print('Test_Acc=[{}]'.format(model.measure(mnist.test_x_set.squeeze(-1), mnist.test_y_set)))


if __name__ == '__main__':
    # mlp_mnist()
    cnn_mnist()
    rnn_mnist()
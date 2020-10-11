# CNN-numpy
## Introduction
Convolutional Neural Networks implemented in NumPy.

## Require  
Python 3  
Numpy  

## Example
1. define a Model class
'''
model = Model(name='model', input_dim=[1, 28, 28])
'''
2. initial the structure
'''
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
'''

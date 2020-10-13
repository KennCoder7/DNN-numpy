# Introduction
Deep Neural Networks implemented in NumPy.  
Including [Convolutional Neural Networks](#CNN-numpy), Multi-Layer Perceptron, and LSTM coming so...

# Require  
Python 3  
Numpy  

# CNN-numpy
Perfrom MNIST recognition task.  
``
C5*3*3-Relu-P2*2-Flatten-FC100-Relu-FC10-Softmax
``  
Test_Acc=[0.9649] after 500 epoches.

## Example
1. define a Model class  
``` 
model = Model(name='model', input_dim=[1, 28, 28])  
```

2. initial the structure  
```
model.initial(
    [
        Conv2D(name='C1', kernel_size=[3, 3], filters=5, padding='valid'),
        Activation(name='A1', method='relu'),
        MaxPooling2D(name='P1', pooling_size=[2, 2]),
        Flatten(name='flatten'),
        Dense(name='fc1', units=100),
        Activation(name='A2', method='relu'),
        Dense(name='fc2', units=10),
        Activation(name='A3', method='softmax'),
    ]
)  
```

3. fit the training set  
```
model.fit(mnist.train_x_set, mnist.train_y_set)
```

4. training, print the training log at every interval epoch.    
```
model.train(lr=0.01, momentum=0.9, max_epoch=500, batch_size=128, interval=10)  
```  

5. print test result  
```
print('Test_Acc=[{}]'.format(model.measure(mnist.test_x_set, mnist.test_y_set)))  
```

## Note
Training requires lots of time.  
The reason why the training accuracy vibrates is about it computes the accuracy of batch set rather than whole training set.    

# MLP-numpy  
Multi-Layer Perceptron implemented in NumPy.   
```
def mlp_mnist():
    mnist = MNIST(dimension=3)
    model = Model(name='model', input_dim=[1, 28, 28])
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
    model.train(lr=0.01, momentum=0.9, max_epoch=500, batch_size=128, interval=10)
    print('Test_Acc=[{}]'.format(model.measure(mnist.test_x_set, mnist.test_y_set)))
  ```  
  

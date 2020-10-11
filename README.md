# CNN-numpy
## Introduction
Convolutional Neural Networks implemented in NumPy.  
Perfrom MNIST recognition task.  
C5*3*3-Relu-P2*2-FC100-Relu-FC10-Softmax  
Test_Acc=[0.9458] after 500 epoches.

## Require  
Python 3  
Numpy  

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
        Activation(name='A3', method='relu'),
        Dense(name='fc2', units=10),
        Activation(name='A4', method='softmax'),
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

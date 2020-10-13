# Introduction
Deep Neural Networks implemented in NumPy.  
Including [Multi-Layer Perceptron](#MLP-numpy), [Convolutional Neural Networks](#CNN-numpy), and LSTM coming so...  
Now bascily preforming classification task, so Cross Entropy Loss and Batch Stochastic Gradient Descent are implemented.
# Require  
Python 3  
Numpy  

# CNN-numpy
## Principle
### Convolutional Block
Forward:  ![](http://latex.codecogs.com/svg.latex?z^l=z^{l-1}*w^l+b^l)  
Backward: ![](http://latex.codecogs.com/svg.latex?e^{l-1}=e^l*rot180(w^l))  
Gradient: ![](http://latex.codecogs.com/svg.latex?dw^l=z^{l-1}*e^l,db^l=sum(e^l))  
Where * is refer to convolution operation.  
### Activation Block
Forward: ![](http://latex.codecogs.com/svg.latex?z^l=f(z^{l-1}))  
Backward: ![](http://latex.codecogs.com/svg.latex?e^{l-1}=df(z^{l-1})*e^l)  
Where f is refer to activation function, df is refer to derivative activation function, and * is refer to multiply operation. 
### MaxPooling Block
Forward: ![](http://latex.codecogs.com/svg.latex?z^l=f(z^{l-1}))  
Backward: ![](http://latex.codecogs.com/svg.latex?e^{l-1}=df(e^l))  
Where f is refer to maxpooling sample, df is refer to upsample.
### Dense Block (Fully Connected Block)
Forward: ![](http://latex.codecogs.com/svg.latex?w^l=w^l*z^{l-1}+b^l)  
Backward: ![](http://latex.codecogs.com/svg.latex?e^{l-1}=(w^l)^T*e^l)  
Gradient: ![](http://latex.codecogs.com/svg.latex?dw^l=z^{l-1}*e^l,db^l=e^l)  
Where * is refer to matmul operation.  
## Example
Perfrom MNIST recognition task.  
``
C5*3*3-Relu-P2*2-Flatten-FC100-Relu-FC10-Softmax
``  
Test_Acc=0.9649 after 500 epoches.  
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
## Principle
### Dense Block (Fully Connected Block)
Forward: ![](http://latex.codecogs.com/svg.latex?w^l=w^l*z^{l-1}+b^l)  
Backward: ![](http://latex.codecogs.com/svg.latex?e^{l-1}=(w^l)^T*e^l)  
Gradient: ![](http://latex.codecogs.com/svg.latex?dw^l=z^{l-1}*e^l,db^l=e^l)  
Where * is refer to matmul operation.  
## Example
Perfrom MNIST recognition task.  
``
Flatten-FC100-Relu-FC100-Relu-FC10-Softmax
``  
Test_Acc=0.9525 after 500 epoches.
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
  

#implementing adaline.
#note that plot code is omitted in p42, 44, 45.

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

CONFIG_ENABLE_PLOT=0

class AdalineSGD:
    '''
    eta-learning rate
    n_iter=number of iteration
    random_state=generator seed for random weight
    '''
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        self.random_state=random_state

    ''' fit training data 
    X: [n_examples, n_features]
    y: [n_examples]
    '''

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_=[]

        for i in range(self.n_iter):
            if self.shuffle:
                X,y=self._shuffle(X,y)
            losses=[]

            counter = 0 
            for xi, target in zip(X,y):
    
                print(f'{counter}:x xi: {xi.shape}, target: {target.shape}')
                losses.append(self._update_weights(xi, target))
                counter+=1

            avg_loss=np.mean(losses)
            self.losses_.append(avg_loss)
        return self            


    ''' fit training data without reinitializint the weights. '''

    def partial_fit(self, X,y):
        if not self.w_.initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
            self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self   

    ''' shuffle training data'''

    def _shuffle(self, X, y):
        r=self.rgen.permutation(len(y))
        return X[r], y[r]

    ''' init weights to small random numbers '''
    
    def _initialize_weights(self, m):
        self.rgen=np.random.RandomState(self.random_state)
        self.w_=self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_=np.float_(0.)
        self.w_initialized=True

    ''' adaline learning rule applied to update the weigths.'''

    def _update_weights(self, xi, target):
        output=self.activation(self.net_input(xi))
        error=(target-output)
        self.w_+=self.eta * 2.0 * xi * (error)
        self.b_+=self.eta * 2.0 * error
        loss=error**2
        return loss

    ''' compute linear activation'''

    def activation(self, X):
        return X

    '''
    Calculate net input
    input: X[100,2], self.w_[2] (x: 100 samples, 2 features, features = weights.
    output: [100]
    
    '''
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    '''
    Return class label after unit step
    Given x if net_input's dot is greater than 0.5, return 1 otherwise 0. (binary class.?)
    '''

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

S='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL: ', S)
df=pd.read_csv(S, header=None, encoding='utf-8')
print(df.tail)

#select setosa and versicolor

y=df.iloc[0:100, 4].values
y=np.where(y=='Iris-setosa', 0, 1)

# extract sepal length and petal length

X=df.iloc[0:100, [0, 2]].values

adaline=AdalineSGD(eta=0.01, n_iter=10)
adaline.fit(X,y)

print(f'X,y shapes: {X.shape}, {y.shape}')
print(f'adaline.losses_: {adaline.losses_}')

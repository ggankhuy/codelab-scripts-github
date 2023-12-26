#perceptron implementation in python

import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter

CONFIG_ENABLE_PLOT=1

class Perceptron:
    '''
    eta-learning rate
    n_iter=number of iteration
    random_state=generator seed for random weight
    '''
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    ''' fit training data 
    X: [n_examples, n_features]
    y: [n_examples]
    '''

    def fit(self, X, y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_=np.float_(0.)
        self.errors_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(X,y):
                update=self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    '''
    Calculate net input
    '''
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    '''
    Return class label after unit step
    Given x if net_input's dot is greater than 0, return 1 otherwise 0. (binary class.?)
    '''

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    
    
        
S='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('From URL: ', S)
df=pd.read_csv(S, header=None, encoding='utf-8')
print(df.tail)

#select setosa and versicolor

y=df.iloc[0:100, 4].values
y=np.where(y=='Iris-setosa', 0, 1)

# extract sepal length and petal length

X=df.iloc[0:100, [0, 2]].values

if CONFIG_ENABLE_PLOT:
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')

    plt.legend(loc='upper left')
    plt.show()

ppn=Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)

print(f'X,y shapes: {X.shape}, {y.shape}')
print(f'ppn.errors_: {ppn.errors_}')
if CONFIG_ENABLE_PLOT:
    plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

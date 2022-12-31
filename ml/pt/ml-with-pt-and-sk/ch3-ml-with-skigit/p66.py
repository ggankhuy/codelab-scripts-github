#implementing adaline.
#note that plot code is omitted in p42, 44, 45.

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONFIG_ENABLE_PLOT=1

class LogisticRegressionGD:
    '''
    eta-learning rate
    n_iter=number of iteration
    random_state=generator seed for random weight
    '''
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
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
        self.losses_=[]

        for i in range(self.n_iter):

            #net_input.shape: [70,]            

            net_input=self.net_input(X)

            #output.shape: [70,]

            output=self.activation(net_input)

            #errors.shape: [70,]

            errors=(y-output)

            # self.w_: [2]
            # self.b_: scalar.

            self.w_+=self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_+=self.eta * 2.0 * errors.mean()

            # loss scalar.
 
            loss=(-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output)))/X.shape[0])
            
            # self.losses_: 1-d array len same as n_iter: 1000.
            self.losses_.append(loss)
        return self            

    ''' compute logistic sigmoid activation'''

    def activation(self, z):
        return 1./(1. + np.exp(-np.clip(z,-250, 250)))
        

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


iris = datasets.load_iris()
X=iris.data[:, [2,3]]
y=iris.target
print('class labels: ', np.unique(y))

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

X_train_01_subset=X_train_std[(y_train==0)|(y_train==1)]
y_train_01_subset=y_train[(y_train==0)|(y_train==1)]
lrgd=LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, clf=lrgd)
if CONFIG_ENABLE_PLOT:
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

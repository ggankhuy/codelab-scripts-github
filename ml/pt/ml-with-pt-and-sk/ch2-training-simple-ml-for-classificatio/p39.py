#implementing adaline.
#note that plot code is omitted in p42, 44, 45.

import numpy as np
import os
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CONFIG_ENABLE_PLOT=0

class AdalineGD:
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
        self.losses_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(X)

            #output.shape: [100]

            output=self.activation(net_input)

            #errors.shape: [100]

            errors=(y-output)

            # self.w_: [2]
            # self.b_: scalar.

            self.w_+=self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_+=self.eta * 2.0 * errors.mean()

            # loss scalar.

            loss=(errors**2).mean()
            
            # self.losses_: 1-d array updated with loss. 

            self.losses_.append(loss)
        return self


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

if CONFIG_ENABLE_PLOT:
    fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    ada1=AdalineGD(n_iter=15, eta=0.1).fit(X,y)
    ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - learning rate 0.1')
    ada2=AdalineGD(n_iter=15, eta=0.0001).fit(X,y)
    ax[1].plot(range(1, len(ada2.losses_) + 1), np.log10(ada2.losses_), marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Mean squared error)')
    ax[1].set_title('Adaline - learning rate 0.0001')
    plt.show()
else:
    adaline=AdalineGD(eta=0.01, n_iter=10)
    adaline.fit(X,y)

    print(f'X,y shapes: {X.shape}, {y.shape}')
    print(f'adaline.losses_: {adaline.losses_}')


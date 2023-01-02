# logistic function 
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X=np.array([1,1.4,2.5])
w=np.array([0.4, 0.3, 0.5])

CONFIG_ENABLE_PLOT=1

def net_input(X,w):
    return np.dot(X,w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X,w):
    z=net_input(X,w)
    return logistic(z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def tanh(z):
    e_p=np.exp(z)
    e_m=np.exp(-z)
    return (e_p-e_m)/(e_p+e_m)

z=np.arange(-5,5,0.005)
log_act=logistic(z)
tanh_act=tanh(z)

if CONFIG_ENABLE_PLOT:
    plt.ylim([-1.5, 1.5])
    plt.xlabel('net input $z$')
    plt.ylabel('activation $\phi(z)$')
    plt.axhline(1, color='black', linestyle=':')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')

    

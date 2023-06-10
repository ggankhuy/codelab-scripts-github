# logistic function 
import numpy as np
import torch 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X=np.array([1,1.4,2.5])
w=np.array([0.4, 0.3, 0.5])

CONFIG_ENABLE_PLOT=1
CONFIG_USE_PYT=1
CONFIG_USE_NUMPY=0

if CONFIG_USE_PYT and CONFIG_USE_NUMPY:
    print("Error can not set both of CONFIG_USE_PYT and CONFIG_USE_NUMPY...")
    exit(0)

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

print("Computing logistic sigmoid and tanh...")
log_act=logistic(z)
tanh_act=tanh(z)

if CONFIG_USE_PYT:
    print("Using torch tanh/sigmoid function.")
    tanh_act=torch.tanh(torch.from_numpy(z))
    log_act=torch.sigmoid(torch.from_numpy(z))
if CONFIG_USE_NUMPY:
    print("Using numpy tanh function.")
    tanh_act=np.tanh(z)

if CONFIG_ENABLE_PLOT:
    plt.ylim([-1.5, 1.5])
    plt.xlabel('net input $z$')
    plt.ylabel('activation $\phi(z)$')
    plt.axhline(1, color='black', linestyle=':')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle=':')
    plt.axhline(-0.5, color='black', linestyle=':')
    plt.axhline(-1, color='black', linestyle=':')
    plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
    plt.plot(z, log_act, linewidth=3, label='logistic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    

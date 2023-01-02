# logistic function 
import numpy as np
X=np.array([1,1.4,2.5])
w=np.array([0.4, 0.3, 0.5])

def net_input(X,w):
    return np.dot(X,w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X,w):
    z=net_input(X,w)
    return logistic(z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

W=np.array([[1.1,1.2,0.8,0.4], [0.2, 0.4, 1.0, 0.2], [0.6, 1.5, 1.2, 0.7]])
A=np.array([[1, 0.1, 0.4, 0.6]])
Z=np.dot(W, A[0])
y_probas = softmax(Z)
print('Probabilities:\n', y_probas)
print('Sum of probabilities: ', np.sum(y_probas))

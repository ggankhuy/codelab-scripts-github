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

print(f'P(y=1|x)={logistic_activation(X,w):.3f}')


import numpy as np
W=np.array([[1.1,1.2,0.8,0.4], [0.2, 0.4, 1.0, 0.2], [0.6, 1.5, 1.2, 0.7]])
A=np.array([[1, 0.1, 0.4, 0.6]])
Z=np.dot(W, A[0])
y_probas = logistic(Z)
print('Net input: \n',Z)
print('Output Units:\n', y_probas)
y_class=np.argmax(Z,axis=0)
print('Predicted class label: ', y_class)


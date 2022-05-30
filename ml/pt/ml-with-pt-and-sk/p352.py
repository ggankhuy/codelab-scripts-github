#from neuralnet import NeuralNetMLP
import neuralnet
import numpy as np
num_epochs=50
minibatch_size=100

# the code along may not execute unless you put add'l driver code.

def minibatch_generator(X, y, minibatch_size):
    DEBUG = 1

    if DEBUG:
        print("X, y, minibatch_size: ", X, y, minibatch_size)
   
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    if DEBUG:
        print("indices.shape: ", indices.shape)
    
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

def sigmoid(z):
    return 1./(1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    ary=np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val]  = 1
    return any

#inheritance not working!
#class NeuralNetMLP(neuralnet):
class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        DEBUG=1
        super().__init__()

        if DEBUG:
            print("num_features, num_hidden, num_classes: ", num_features, num_hidden, num_classes)

        self.num_classes = num_classes
        
        # hidden

        rng=np.random.RandomState(random_seed)

        if DEBUG:
            print("rng: ", rng)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        if DEBUG:
            print("weight_h: ", self.weight_h.shape)
            print("bias_h: ", self.bias_h.shape)

        # output

        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

        if DEBUG:
            print("weight_out: ", self.weight_out.shape)
            print("bias_out: ", self.bias_out.shape)

    def forward(self, X):
        DEBUG=1
        if DEBUG:
            print("forward entered...")
        
        # Hidden layer

        # input dim: [n_hidden, n_features]
        # dot [n_featues, n_examples].T
        # output dim: [n_examples, n_hidden]

        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        if DEBUG:
            print("z_h: ", z_h.shape)
            print("a_h: ", a_h.shape)

        # Output layer

        # input dim: [n_classes, n_hidden]
        # dot [n_hidden, n_examples].T
        # output dim: [n_examples, n_hidden]

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out 
        a_out = sigmoid(z_out)

        if DEBUG:
            print("z_out: ", z_out.shape)
            print("a_out: ", a_out.shape)

        return a_h, a_out

    def backward(self, X, a_h, a_out, y):
        
        #######################
        # output layer weights 
        #######################

        # one-hot encoding

        y_onehot = int_to_onehot(y, self.num_classes)

        if DEBUG:
            print("y_onehot: ", y_onehot)

        # Part 1: dLoss / dOutWeights 

        ## = dLoss / dOutact * dOutAct/dOutNet * dOutNet/dOutWeight 
        ## where DeltaOut = dLoss/dOutAct * dOutAct / dOutnNet
        ## for convenient re-use

        # intput/output dim: [n_examples, n_classes]
        
        d_loss__d_a_out = 2. *(a_out - y_onehot) / y/shape[0]

        # input/output dim: [n_examples, n_classes]

        d_a_out__d_z_out = a_out * (1.-a_out)
        
        # output dim: [n_examples, n_classes]

        delta_out = d_loss_d_a_out * d_a_out__d_z_out

        # gradient for output weights

        # [n_examples, n_hidden]

        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples]
        # dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]

        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        ############################

        # part 2: dLoss/dHiddenWeights 
        ## =  DeltaOut * dOutNet / dHiddenAct * dHiddenAct/dHiddenNet
        # * dHiddenNet/dWeight

        # [n_classes, n_hidden]

        d_z_out__a_h = self.weight_out
        
        # output dim: [n_examples, n_hidden]

        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_features]

        # output dim: [n_hidden, n_features]

        d_loss_d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss_d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_oiut, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
    
model=NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)
print(model)

# following prints not working because inheritance of neuralnet class not working!
#print(model.summary)
#print(model.layers)

# iterate over training epochs

'''
# not working beause x_train, y_train not init-d.
for i in range(num_epochs):

    # iterate over minibatches

    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

    print("X_train_mini.shape: ", X_train_mini.shape)        
    print("y_train_mini.shape: ", y_train_mini.shape)
'''

def mse_loss(targets, probas, num_labels=10):
    onehost_targets - int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehost_targets - probas) ** 2 )

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)




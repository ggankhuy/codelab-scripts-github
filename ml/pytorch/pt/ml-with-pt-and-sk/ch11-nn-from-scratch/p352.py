#from neuralnet import NeuralNetMLP
#import neuralnet
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DEBUG=0
num_epochs=50
minibatch_size=100

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X=X.values
y=y.astype(int).values
print(X.shape)
print(y.shape)

X = ((X/255.)-0.5)*2

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax=ax.flatten()

'''
for i in range(10):
    img = X[y==1][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
'''
X_temp, X_test, y_temp, t_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

if DEBUG:
    print("X_test, X_train, X_valid shapes: ", X_test.shape, X_train.shape, X_valid.shape)
    print("y_train, y_valid shapes: ", y_train.shape, y_valid.shape)

# done with from p344.py

# the code along may not execute unless you put add'l driver code.

def minibatch_generator(X, y, minibatch_size):
    DEBUG = 0

    if DEBUG:
        print("X, y, minibatch_size: ", X.shape, y.shape, minibatch_size)
   
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
    return ary

#inheritance not working!
#class NeuralNetMLP(neuralnet):
class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        DEBUG=0
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

    def forward(self, x):
        DEBUG=0
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
            print("y_onehot.shape: ", y_onehot.shape)

        # Part 1: dLoss / dOutWeights 

        ## = dLoss / dOutact * dOutAct/dOutNet * dOutNet/dOutWeight 
        ## where DeltaOut = dLoss/dOutAct * dOutAct / dOutnNet
        ## for convenient re-use

        # intput/output dim: [n_examples, n_classes]
        
        d_loss__d_a_out = 2. *(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]

        d_a_out__d_z_out = a_out * (1.-a_out)
        
        # output dim: [n_examples, n_classes]

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

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

        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative.

        d_z_h__d_w_h = X

        # output dim: [n_hidden, n_features]

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)

def mse_loss(targets, probas, num_labels=10):
    DEBUG=0
    if DEBUG:
        print("mse_loss: entered...")
        print("targets.shape, probas.shape, num_label: ", targets.shape, probas.shape, num_labels)

    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    print("onehot_targets.shape: ", onehot_targets.shape)

    return np.mean((onehot_targets - probas) ** 2 )

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels = num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss

    mse  = mse / i
    acc = correct_pred / num_examples 
    return mse, acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    DEBUG=0
    epoch_loss=[]
    epoch_train_acc=[]
    epoch_valid_acc=[]

    for e in range(num_epochs):
        # iterate over minibatches

        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)


        for X_train_mini, y_train_mini in minibatch_gen:
            if DEBUG:
                print("X_train_mini, y_train_mini shapes: ", X_train_mini.shape, y_train_mini.shape)

            # compute outputs.

            a_h, a_out = model.forward(X_train_mini)

            # compute gradients.

            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)
    
            # update weights.

            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        
        # epoch logging. 

        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)

        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d}'
                f'| Train MSE: {train_mse:.2f} '
                f'| Train Acc: {train_acc:.2f} '
                f'| Valid Acc: {train_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc

            
model=NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)
print(model)

# following prints not working because inheritance of neuralnet class not working!
#print(model.summary)
#print(model.layers)

# iterate over training epochs

# not working beause x_train, y_train not init-d.
'''
for i in range(num_epochs):

    # iterate over minibatches

    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break

    print("X_train_mini.shape: ", X_train_mini.shape)        
    print("y_train_mini.shape: ", y_train_mini.shape)

'''
'''
_, probas = model.forward(X_valid)


print("probas: ", probas.shape)
mse=mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')
predicted_labels = np.argmax(probas, axis=1)
acc=accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')

'''
'''
mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valida accuracy: {acc*100:.1f}%')
'''
np.random.seed(123)

if DEBUG:
    print("X_test, X_train, X_valid shapes: ", X_test.shape, X_train.shape, X_valid.shape)
    print("y_train, y_valid shapes: ", y_train.shape, y_valid.shape)

epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, \
    num_epochs = 50, learning_rate=0.1)


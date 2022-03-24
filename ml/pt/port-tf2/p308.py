# using neural net for regression task.

import torch
import torch.nn as nn
import helper
import sys
import time
import re 
import numpy as np
import matplotlib as plt

from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

DEBUG=0
TEST=1
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONFIG_EPOCHS=30
CONFIG_BATCH_SIZE=100

for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("epochs", i):
            CONFIG_EPOCHS=int(i.split('=')[1])

        if re.search("batch_size", i):
            CONFIG_BATCH_SIZE=int(i.split('=')[1])

    except Exception as msg:
        print(msg)
        print("No argument provided, default values will be used.")

print("epochs: ", CONFIG_EPOCHS)
print("batch_size: ", CONFIG_BATCH_SIZE)
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};

def prepare_data():
    housing = fetch_california_housing()
    print("housing.data/housing.target (type/dtype/len): ", \
        type(housing.data),housing.data.dtype, housing.data.shape, \
        type(housing.target), housing.target.dtype, housing.target.shape)

    train_full, test, train_target_full, test_target = train_test_split(housing.data, housing.target)
    train, valid, train_target, valid_target = train_test_split(train_full, train_target_full)

    print("train/valid/test: ", type(train), train.shape, type(valid), valid.shape, type(test), test.shape)
    print("train/valid/test(targets): ", type(train_target), train_target.shape, \
        type(valid_target), valid_target.shape, \
        type(test_target), test_target.shape)

    train_dl = torch.utils.data.DataLoader(train, batch_size=CONFIG_BATCH_SIZE, shuffle=True)
    train_target_dl = torch.utils.data.DataLoader(train_target, batch_size=CONFIG_BATCH_SIZE, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    test_target_dl = torch.utils.data.DataLoader(test_target, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(valid, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    valid_target_dl = torch.utils.data.DataLoader(valid_target, batch_size=CONFIG_BATCH_SIZE, shuffle=False)

    print("train_dl/valid_dl/test_dl(targets): ", type(train_target_dl), len(train_target_dl), type(valid_target_dl), len(valid_target_dl), type(test_target_dl), len(test_target_dl))
    return [train_dl, train_target_dl], [valid_dl, valid_target_dl], [test_dl, test_target_dl]

class MLP(Module):

    # define model elements

    def __init__(self):
        super(MLP, self).__init__()
        #self.flatten = nn.Flatten(1, 3)

        self.hidden1 = Linear(8, 300).double()

        if DEBUG:
            print("hidden1 dtype: ", self.hidden1.weight.dtype)

        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(300, 1).double()
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
 
    # forward propagate input
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        return X

# train the model

def train_model(train_dl, model):
    # define the optimization
    print("training...")
    print("model parameters: ", model.parameters())
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad()

    # enumerate epochs

    for epoch in range(CONFIG_EPOCHS):
        print("epoch:", epoch, "/", CONFIG_EPOCHS, end="")

        # enumerate mini batches

        i=0
        for inputs, targets in zip(enumerate(train_dl[0]), enumerate(train_dl[1])):
            if i % 20 == 0:
                print(".", end="", flush=True)
    
            inputs=inputs[1]
            targets=targets[1]
            inputs.double()
            targets.double()

            if DEBUG:
                #print("\ninputs: ", type(inputs), inputs.size(), inputs.type(), "\n", inputs)
                #print("targets: ", type(targets), len(targets), targets.type(), "\n", targets)
                print("\ninputs: ", type(inputs), inputs.size())
                print("targets: ", type(targets), targets.size())

            # compute the model output

            yhat = model(inputs[1])

            if DEBUG:
                print("yhat: ", yhat.size())
                print("targets: ", targets.size())

            # calculate loss

            if DEBUG:
                print("\nyhat: ", type(yhat), yhat.size(), yhat.type(), "\n", yhat)
                print("targets: ", type(targets), len(targets), targets.type(), "\n", targets)

            loss = criterion(yhat, targets[1])
            loss.backward()
            # update model weights
            optimizer.step()
            i+=1
        print("loss: ", loss)
        
 
# evaluate the model
def evaluate_model(test_dl, model):
  
    print("evaluating...")
    predictions, actuals = list(), list()

    i=0
    for inputs, targets in zip(enumerate(valid_dl[0]), enumerate(valid_dl[1])):
        if i % 20 == 0:
            print(".", end="", flush=True)

        inputs=inputs[1]
        targets=targets[1]
        inputs.double()
        targets.double()

        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
 
# prepare the data
#path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, valid_dl, test_dl = prepare_data()
# define the network
model = MLP()
# train the model
print("train_dl: ", len(train_dl))
train_model(train_dl, model)
# evaluate the model
evaluate_model(valid_dl, model)
torch.save(model, "p308.h5")

print("predicting...")

i=0
predictions, actuals = list(), list()
for inputs, targets in zip(enumerate(test_dl[0]), enumerate(test_dl[1])):
    if i % 20 == 0:
        print(".", end="", flush=True)

    inputs=inputs[1]
    targets=targets[1]
    inputs.double()
    targets.double()

    if DEBUG:
        print("inputs: ", inputs, type(inputs), inputs.size())
        print("targets: ", targets, type(targets), targets.size())

    # evaluate the model on the test set
    yhat = model(inputs)
    if  DEBUG:
        print("yhat: ", yhat, type(yhat), yhat.size())
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    actual = targets.numpy()
    # convert to class labels
    yhat = argmax(yhat, axis=1)
    # reshape for stacking
    actual = actual.reshape((len(actual), 1))
    yhat = yhat.reshape((len(yhat), 1))
    # store
    predictions.append(yhat)
    actuals.append(actual)
    i+=1
predictions, actuals = vstack(predictions), vstack(actuals)
print("predictions: ", predictions[:10])
print("actuals: ", actuals[:10])


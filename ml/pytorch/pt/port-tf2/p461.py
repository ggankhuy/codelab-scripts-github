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
from torch.nn import Conv2d
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

DEBUG=0
TEST=0
DEBUG_PRT=0
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

CONFIG_EPOCHS=10
CONFIG_BATCH_SIZE=64

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

if TEST:
    CONFIG_BATCH_SIZE=10
    CONFIG_EPOCHS=1
print("epochs: ", CONFIG_EPOCHS)
print("batch_size: ", CONFIG_BATCH_SIZE)

def printdbg(msg):
   if DEBUG_PRT:
        print(msg)

def prepare_data():
    train = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train, valid = torch.utils.data.random_split(train, [50000, 10000])

    print("train/valid/test: ", type(train), len(train), type(valid), len(valid), type(test), len(test))
    train_dl = torch.utils.data.DataLoader(train, batch_size=CONFIG_BATCH_SIZE, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    valid_dl = torch.utils.data.DataLoader(valid, batch_size=CONFIG_BATCH_SIZE, shuffle=False)
    print("train_dl/test_dl: ", type(train_dl), len(train_dl), type(test_dl), len(test_dl))

    return train_dl, valid_dl, test_dl

class MLP(Module):

    # define model elements

    def __init__(self):
        super(MLP, self).__init__()        
        self.conv1 = Conv2d(1, 64, 7, padding="same")
        self.act1 = ReLU()
        self.maxpool1 = MaxPool2d(2)

        self.conv2a = Conv2d(64, 128, 3, padding="same")
        self.act2a = ReLU()
        self.conv2b = Conv2d(128, 128, 3, padding="same")
        self.act2b = ReLU()
        self.maxpool2 = MaxPool2d(2)

        self.conv3a = Conv2d(128, 256, 3, padding="same")
        self.act3a = ReLU()
        self.conv3b = Conv2d(256, 256, 3, padding="same")
        self.act3b = ReLU()
        self.maxpool3= MaxPool2d(2)

        self.flatten = nn.Flatten(1, 3)

        self.hidden1 = Linear(2304, 128)
        self.drop1 = Dropout()
        self.hidden2 = Linear(128, 64)
        self.drop2 = Dropout()
        self.hidden3 = Linear(64, 10)
        self.act3c = Softmax()
        
    # forward propagate input

    def forward(self, X):
        if DEBUG:
            print("forward entered: X: ", X.size()) 

        printdbg("X: " + str(X.size()))

        X = self.conv1(X)
        X = self.act1(X)
        printdbg("X, conv1/act1: " + str(X.size()))
        X = self.maxpool1(X)
        printdbg("X, maxpool1: " + str(X.size()))

        X = self.conv2a(X)
        X = self.act2a(X)
        printdbg("X, conv2a/act2a: " + str(X.size()))
        X = self.conv2b(X)
        X = self.act2b(X)
        printdbg("X, conv2b/act2b: " + str(X.size()))
        X = self.maxpool2(X)
        printdbg("X, maxpool2: " + str(X.size()))
        X = self.conv3a(X)
        X = self.act3a(X)
        printdbg("X, conv3a/act3a: " + str(X.size()))
        X = self.conv3b(X)
        X = self.act3b(X)
        printdbg("X, conv3b/act3b: " + str(X.size()))
        X = self.maxpool3(X)
        printdbg("X, maxpool3: " + str(X.size()))

        X = self.flatten(X)
        printdbg("X, flatten: " + str(X.size()))

        X = self.hidden1(X)
        printdbg("X, hidden1: " + str(X.size()))
        X = self.drop1(X)
        printdbg("X, drop1: " + str(X.size()))
        X = self.hidden2(X)
        printdbg("X, hidden2: " + str(X.size()))
        X = self.drop2(X)
        printdbg("X, drop2: " + str(X.size()))
        X = self.hidden3(X)
        printdbg("X, hidden2: " + str(X.size()))
        X = self.act3c(X)
        printdbg("forward: X (returned): " + str(X.size())) 
        return X

# train the model
def train_model(train_dl, model):

    # define the optimization.
    # refer to: https://stackoverflow.com/questions/63403485/is-there-a-version-of-sparse-categorical-cross-entropy-in-pytorch

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs.

    for epoch in range(CONFIG_EPOCHS):
        print("epoch:", epoch, "/", CONFIG_EPOCHS, end="")

        # enumerate mini batches.

        for i, (inputs, targets) in enumerate(train_dl):

            if TEST:
                if i > 2:
                    quit(0)

            if i % 20 == 0:
                print(".", end="", flush=True)

            if DEBUG or TEST:
                print("\ninputs: ", type(inputs), inputs.size(), inputs)
                print("targets: ", type(targets), targets.size(), targets)

            # clear the gradients.

            optimizer.zero_grad()

            # compute the model output.

            yhat = model(inputs)

            if DEBUG:
                print("yhat: ", yhat.size())
                print("targets: ", targets.size())

            # calculate loss.

            if DEBUG:
                print("yhat: ", type(yhat), yhat.shape)
                print("targets:   ", type(targets), targets.shape)

            loss = criterion(yhat, targets)

            # credit assignment

            loss.backward()

            # update model weights

            optimizer.step()
        print("loss: ", loss)    
        
 
# evaluate the model

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()

    for i, (inputs, targets) in enumerate(test_dl):

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

    acc = accuracy_score(actuals, predictions)
    return acc
 
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
# path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'

train_dl, valid_dl, test_dl = prepare_data()

print(len(train_dl.dataset), len(test_dl.dataset))

# define the network

model = MLP()

# train the model

print("train_dl: ", len(train_dl))
train_model(train_dl, model)

# evaluate the model

acc = evaluate_model(valid_dl, model)
print('Accuracy: %.3f' % acc)

# make a single prediction

print("Making prediction...")

enum_test_dl = list(enumerate(test_dl))
enum_test_dl_sub=enum_test_dl[:1]
print("enum_test_dl_sub: ", len(enum_test_dl_sub))
yhat = model(enum_test_dl_sub[0][1][0])
yhat = yhat.detach().numpy()
actual = enum_test_dl_sub[0][1][1].numpy()

# convert to class labels

yhat = argmax(yhat, axis=1)

# reshape for stacking
#actual = actual.reshape((len(actual), 1))
#yhat = yhat.reshape((len(yhat), 1))

print("yhat:   ", yhat[:10])
print("actual: ", actual[:10])

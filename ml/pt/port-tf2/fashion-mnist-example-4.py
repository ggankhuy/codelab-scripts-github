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
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

DEBUG=0
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

print("epochs: ", CONFIG_EPOCHS)
print("batch_size: ", CONFIG_BATCH_SIZE)
labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};

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
# prepare the dataset
    # define model elements
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten(1, 3)
        # input to first hidden layer
        self.hidden1 = Linear(784, 300)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(300, 100)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(100, 30)
#        xavier_uniform_(self.hidden3.weight)
#        self.act3 = Softmax(dim=1)
        self.act3 = Softmax()
 
    # forward propagate input
    def forward(self, X):

        if DEBUG:
            print("forward entered: X: ", X.size()) 
        # input to first hidden layer
        X = self.flatten(X)

        if DEBUG:
            print("forward: X (flatten): ", X.size()) 

        X = self.hidden1(X)

        if DEBUG:
            print("forward: X (hidden1): ", X.size()) 

        X = self.act1(X)

        if DEBUG:
            print("forward: X (act1/RELU): ", X.size()) 

        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
#        X = self.act3(X)

        if DEBUG:
            print("forward: X (returned): ", X.size()) 

        return X

# train the model
def train_model(train_dl, model):
    # define the optimization

    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # enumerate epochs

    for epoch in range(CONFIG_EPOCHS):
        print("epoch:", epoch, "/", CONFIG_EPOCHS, end="")

        # enumerate mini batches

        for i, (inputs, targets) in enumerate(train_dl):
            if i % 20 == 0:
                print(".", end="", flush=True)

            if DEBUG:
                print("inputs: ", inputs.size())
                print("targets: ", targets.size())

            # clear the gradients

            optimizer.zero_grad()

            # compute the model output

            yhat = model(inputs)
            if DEBUG:
                print("yhat: ", yhat.size())
                print("targets: ", targets.size())
            # calculate loss
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
#path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
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
#print("actual1: ", actual)
# convert to class labels
yhat = argmax(yhat, axis=1)
# reshape for stacking
#actual = actual.reshape((len(actual), 1))
#yhat = yhat.reshape((len(yhat), 1))
print("yhat:   ", yhat[:10])
print("actual: ", actual[:10])

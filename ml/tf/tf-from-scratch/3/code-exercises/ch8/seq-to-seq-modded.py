# data format
# points, direction 
# points=256,4,2, direction256.
# points=[number of samples, four corners, coordinates]
# directions=[directions: 0 or 1]

import copy 
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,  random_split, TensorDataset
import sys
sys.path.append('..')
from common.settings import *
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep
from plots.chapter8 import plot_data
points, directions = generate_sequences(256,  seed=13)


# create rnn_cell.
# rnn_state: 
# - weight_ih: [n_features, hidden_dim], bias_ih: [hidden_dim]
# - weight_hh: [hidden_dim, hidden_dim], bias_hh: [hidden_dim]

#n_features=2
#hidden_dim=2
torch.manual_seed(19)
rnn_cell=nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state=rnn_cell.state_dict()
printDbg(rnn_state)

# Lets do above RNNcell manually. However we are copying the values of weigts, bias initialized values
# so that output can be compared for sanity check!

class Shark:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class RNNCell:
    def __init__(self, rnn_cell_src: nn.RNNCell, input_size:int, hidden_size:int):
        self.linear_input=nn.Linear(input_size, hidden_size)
        self.linear_hidden=nn.Linear(hidden_size, hidden_size)
        rnn_state=rnn_cell_src.state_dict()

        with torch.no_grad():
            self.linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
            self.linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
            self.linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
            self.linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

        self.initial_hidden=torch.zeros(1, hidden_size)
        printDbg(self.initial_hidden)
        self.th=self.linear_hidden(self.initial_hidden)
        printDbg(self.th)
        self.tx=0
        self.th=0
        self.final_hidden=0

    def forward(self, x):
        self.tx=self.linear_input(x)
        adding=self.th+self.tx
        printDbg(adding)
        self.final_hidden=torch.tanh(adding)

X=torch.as_tensor(points[0]).float()
print(X)

rnn_cell_manual=RNNCell(rnn_cell_src=rnn_cell, input_size=n_features, hidden_size=hidden_dim)
print(rnn_cell_manual.forward(X[0:1]))

exit(0)

tx=linear_input(X[0:1])
print(tx)

adding=th+tx
print(adding)

print("Output of our manually written RNN cell:")
print(torch.tanh(adding)) 

print("Output of RNN cell from library:")
print(rnn_cell(X[0:1]))

#p114.
# X.shape[0] = [4,2]

print("running rnn_cell on X[0]:")
hidden = torch.zeros(1, hidden_dim)
for i in range(X.shape[0]):
    out = rnn_cell(X[i:i+1], hidden)
    print(out)
    hidden=out

print("creating nn.RNN equivalent of above")    

torch.manual_seed(19)
rnn=nn.RNN(input_size=n_features, hidden_size=hidden_dim)
print(rnn.state_dict())

# p119

# RNN input shape (default):    LNF= length, batch, features
# RNN input shape (batch1st):   NLF= batch, length, features
# RNN input shape (packaed seq): (later)

# Simple RNN init hidden state:     1NH => len=1, batch, hidden_dim
# stacked RNN                       (no. of layers, batch, hidden_dim)
# bidir RNN                         (2 * no. of layers, batch, hidden_dim)

# output 
# Simple RNN output shape (default):    LNH=length, batch, hidden_dim
# batch first                           NLH
# bidir RNN                             L,N,2*H


# input -> output
# (default):        LNF => LNH  matmul([L,N,F][L,N,H])?
# (batch1st):       NLF => NLH  matmul([N,L,F][N,L,H])?

print("computing first 3 data points: 3,4,2...")
batch = torch.as_tensor(points[:3]).float()
print("batch.shape: \n", batch.shape)
# 3,4,2 => batch, sequence, features

permuted_batch=batch.permute(1,0,2)
print("pernuted_batch: \n", permuted_batch)
# 4,3,2

torch.manual_seed(19)
rnn=nn.RNN(input_size=n_features, hidden_size=hidden_dim)
out,final_hidden=rnn(permuted_batch)
print("out:\n", out.shape, out)
print("final_hidden:\n", final_hidden.shape, final_hidden)

batch_hidden=final_hidden.permute(1,0,2)
print("batch_hidden:\n", batch_hidden.shape, batch_hidden)

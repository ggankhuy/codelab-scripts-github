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
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep
from plots.chapter8 import plot_data
points, directions = generate_sequences(256,  seed=13)


# create rnn_cell.
# rnn_state: 
# - weight_ih: [n_features, hidden_dim], bias_ih: [hidden_dim]
# - weight_hh: [hidden_dim, hidden_dim], bias_hh: [hidden_dim]

n_features=2
hidden_dim=2
torch.manual_seed(19)
rnn_cell=nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)
rnn_state=rnn_cell.state_dict()
print(rnn_state)

# Lets do above RNNcell manually. However we are copying the values of weigts, bias initialized values
# so that output can be compared for sanity check!

linear_input=nn.Linear(n_features, hidden_dim)
linear_hidden=nn.Linear(hidden_dim, hidden_dim)

with torch.no_grad():
    linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
    linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
    linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
    linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

initial_hidden=torch.zeros(1, hidden_dim)
print(initial_hidden)
th=linear_hidden(initial_hidden)
print(th)

X=torch.as_tensor(points[0]).float()
print(X)

tx=linear_input(X[0:1])
print(tx)

adding=th+tx
print(adding)

print("Output of our manually written RNN cell:")
print(torch.tanh(adding)) 

print("Output of RNN cell from library:")
print(rnn_cell(X[0:1]))

    
    






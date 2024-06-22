# ch8-p96 rnn_cell implementation with minimum modification (mod only includes extra
# print statements similar mods)

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
from common.settings_modded import *
from common.classes import *
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep
from plots.chapter8 import plot_data
points, directions = generate_sequences(256,  seed=13)

# create rnn_cell.
# rnn_state: 
# - weight_ih: [n_features, hidden_dim], bias_ih: [hidden_dim]
# - weight_hh: [hidden_dim, hidden_dim], bias_hh: [hidden_dim]

print("Import setings:")
printDbg("hidden_dim: ", hidden_dim)
printDbg("n_features: ", n_features)

torch.manual_seed(19)
rnn_cell=nn.RNNCell(input_size=n_features, hidden_size=hidden_dim)

if rnn_cell == None:
    print("rnn_cell is None.")
    quit(1)

rnn_state=rnn_cell.state_dict()

linear_input=nn.Linear(n_features, hidden_dim)
linear_hidden=nn.Linear(hidden_dim, hidden_dim)

with torch.no_grad():
    linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
    linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
    linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
    linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

initial_hidden=torch.zeros(1, hidden_dim)
printTensor(initial_hidden, globals())

th=linear_hidden(initial_hidden)
printTensor(th, globals())

X=torch.as_tensor(points[0]).float()
printTensor(X, globals())

tx=linear_input(X[0:1])
printTensor(tx, globals())

adding=th+tx
printTensor(adding, globals())
printDbg("Adding after torch.tanh()")
adding=torch.tanh(adding)
printTensor(adding, globals())

X_01=X[0:1]
printDbg("X[0:1]: ")
printTensor(X_01, globals())
rnn_cell_output=rnn_cell(X[0:1])
printTensor(rnn_cell_output, globals())



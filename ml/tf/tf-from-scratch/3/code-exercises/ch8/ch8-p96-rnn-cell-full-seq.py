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

printDbg("nn.RNNCell (library):")
for k, v in rnn_state.items():
    printDbg(k, np.array(v).shape, "\n", v)

# Lets do above RNNcell manually. However we are copying the values of weigts, bias initialized values
# so that output can be compared for sanity check!

hidden=torch.zeros(1, hidden_dim)
printTensor(hidden, globals())

X=torch.as_tensor(points[0]).float()
print(X)
printTensor(X, globals())

for i in range(X.shape[0]):
    printDbg("iter: ", i)
    out = rnn_cell(X[i:i+1])
    printTensor(out,globals())
    final_hidden = out

printTensor(final_hidden, globals())

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
from common.classes import *
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep
from plots.chapter8 import plot_data
points, directions = generate_sequences(256,  seed=13)

printTensor(points, globals(), "brief")
printTensor(directions, globals(), "brief")

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

X=torch.as_tensor(points[0]).float()
printTensor(X, globals())

x0=X[0:1]
printDbg("x[0]: ")
printTensor(x0, globals())
rnn_cell_output=rnn_cell(x0)
printTensor(rnn_cell_output, globals())



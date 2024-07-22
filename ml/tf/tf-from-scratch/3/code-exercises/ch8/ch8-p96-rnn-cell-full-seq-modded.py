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
import matplotlib.pyplot as plt

points, directions = generate_sequences(256,  seed=13)

# create rnn_cell.
# rnn_state: 
# - weight_ih: [n_features, hidden_dim], bias_ih: [hidden_dim]
# - weight_hh: [hidden_dim, hidden_dim], bias_hh: [hidden_dim]

CONFIG_ENABLE_PLOT=1
DEBUG=0

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

X=torch.as_tensor(points[0]).float()
print(X)
print("X.shape: ", np.array(X).shape)

rnn_cell_manual=\
    RNNCell(\
    rnn_cell_src=rnn_cell, \
    input_size=n_features, \
    hidden_size=hidden_dim)

printDbg("nn.RNNCell (manual):")

hidden=torch.zeros(1, hidden_dim)

if CONFIG_ENABLE_PLOT:
    fig, axs = plt.subplots(2,2)
    hidden_acc=[]
    fig.suptitle('hidden states during iterations.')

for i in range(X.shape[0]):
    printDbg("iter: ", i)

    out = rnn_cell(X[i:i+1])
    out_manual = rnn_cell_manual(X[i:i+1])

    printTensor(out,globals(), "full")
    printTensor(out_manual,globals(), "full")

    hidden = out
    hidden_manual = out_manual

    if CONFIG_ENABLE_PLOT:
        hidden_acc+=hidden
        print("subplot indices: ", int(i/2), i%2)
        axs[int(i/2), i%2].set_xlim([-1, 1])
        axs[int(i/2), i%2].set_ylim([-1, 1])

        if DEBUG:
            print("hidden.detach().numpy()[0]: ", hidden.detach().numpy()[0])

        x=round(hidden.detach().numpy()[0][0], 2)
        y=round(hidden.detach().numpy()[0][1], 2)
        print("x/y: ", x,y)
        axs[int(i/2), i%2].plot(x,y, marker=8)
        axs[int(i/2), i%2].text(x,y+0.5, '({:2.2}, {:2.2})'.format(x, y))

final_hidden=hidden
final_hidden_manual=hidden_manual

if CONFIG_ENABLE_PLOT:
    plt.show()

printTensor(final_hidden, globals())
printTensor(final_hidden_manual, globals())

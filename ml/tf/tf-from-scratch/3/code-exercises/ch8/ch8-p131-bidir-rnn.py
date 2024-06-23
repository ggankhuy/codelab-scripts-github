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

torch.manual_seed(19)
rnn_bidirect=nn.RNN(input_size=n_features, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
state=rnn_bidirect.state_dict()

for k, v in state.items():
    printDbg("k/v:", k, v)


rnn_forward=nn.RNN(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
rnn_reverse=nn.RNN(input_size=n_features, hidden_size=hidden_dim, batch_first=True)

rnn_forward.load_state_dict(dict(list(state.items())[:4]))
rnn_reverse.load_state_dict(dict([(k[:-8], v) \
    for k, v in list(state.items()) [4:]]
))

x=torch.as_tensor(points[0:1]).float()
x_rev=torch.flip(x,dims=[1])
printTensor(x_rev, globals())

out, h = rnn_forward(x)
out_rev, h_rev = rnn_reverse(x_rev)
out_rev_back = torch.flip(out_rev, dims=[1])

printTensor(out, globals())
printTensor(out_rev, globals())
printTensor(out_rev_back, globals())

out_rev_sum = torch.cat([out, out_rev_back], dim=2)
printTensor(out_rev_sum, globals())

out, hidden = rnn_bidirect(x)
printTensor(out, globals())
printTensor(hidden, globals())

printDbg(out[:, -1] == hidden.permute(1,0,2).view(1, -1))


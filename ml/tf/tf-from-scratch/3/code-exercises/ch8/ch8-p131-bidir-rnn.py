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

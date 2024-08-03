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
print("1. permuted data, using nn.RNN")

batch = torch.as_tensor(points[:3]).float()

printTensor(batch, globals())
permuted_batch=batch.permute(1,0,2)
printTensor(permuted_batch, globals())

torch.manual_seed(19)
rnn=nn.RNN(input_size=n_features, hidden_size=hidden_dim)

out, final_hidden=rnn(permuted_batch)

printTensor(out, globals())
printTensor(final_hidden, globals())

batch_hidden=final_hidden.permute(1,0,2)

printTensor(batch_hidden, globals(), "full")

print("BATCH first...")
print("computing first 3 data points: 3,4,2...")
printTensor(batch, globals())

# [3,4,2] = [batch, length, feature] to [4,3,2] = [Length, batch, feature]
#permuted_batch=batch.permute(1,0,2)

torch.manual_seed(19)
rnn=nn.RNN(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
out, final_hidden=rnn(batch)
printTensor(out, globals())
printTensor(final_hidden, globals(), "full")


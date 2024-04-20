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

#p115.
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

'''
X=torch.as_tensor(points[0]).float()
print(X)

rnn_cell_manual=RNNCell(rnn_cell_src=rnn_cell, input_size=n_features, hidden_size=hidden_dim)

printDbg("nn.RNNCell (manual):")

r1=rnn_cell_manual(X[0:1])
print("Output of our manually written RNN cell:")
print("r1: ", r1)

r2=rnn_cell(X[0:1])
print("Output of RNN cell from library:")
print("r2: ", r2)


exit(0)
'''


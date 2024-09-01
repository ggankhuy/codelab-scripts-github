# points, direction 
# points=256,4,2, direction256.
# points=[number of samples, four corners, coordinates]
# directions=[directions: 0 or 1]

# values are partially correct (1st row correct, 2nd one not), possibly because of that:
# query context_vector and concatenated is not printing same values as book.

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
from plots.chapter9 import sequence_pred
import matplotlib.pyplot as plt
PRINT_TENSOR_OUTPUT="full"

N_FEATURES=global_vars['n_features'] 
HIDDEN_DIM=global_vars['hidden_dim']

def calc_alphas(ks,q):
    N,L,H=ks.size()
    alphas=torch.ones(N,1,L).float()*1/L
    return alphas

# Once enabled, full_seq is created with random values with overriddable settings is applied from global_var.
# If not random, since full_seq uses 2 for feature and hidden_dim, overridable settings in global_var can no longer 
# b3 used.

CONFIG_RANDOM_INIT=0

# [rectangle=1, corners=4, coordinates=2]

if CONFIG_RANDOM_INIT:
    print("Full_seq: random init...")
    full_seq=torch.rand(1,4,5)
else:
    N_FEATURES=2
    HIDDEN_DIM=2
    full_seq=(torch.tensor([[-1,-1],[1,-1], [1,1],[1,-1]]).float().view(1,4,2))

source_seq = full_seq[:, :2]
target_seq = full_seq[:, 2:]

printTensor(full_seq, globals(), PRINT_TENSOR_OUTPUT) 
printTensor(source_seq, globals(), PRINT_TENSOR_OUTPUT) 
printTensor(target_seq, globals(), PRINT_TENSOR_OUTPUT) 

torch.manual_seed(21)
encoder=Encoder(n_features=N_FEATURES, hidden_dim=HIDDEN_DIM)
rnn_state=encoder.state_dict()

printDbg("nn.RNNCell (library):")
for k, v in rnn_state.items():
    printDbg(k, np.array(v).shape, "\n", v)


hidden_seq=encoder(source_seq)
values=hidden_seq
keys=hidden_seq
printTensor(values, globals(), PRINT_TENSOR_OUTPUT)
#exprintTensor(keys, globals(), PRINT_TENSOR_OUTPUT)

torch.manual_seed(21)
decoder=Decoder(N_FEATURES, hidden_dim=HIDDEN_DIM)
decoder.init_hidden(hidden_seq)
inputs=source_seq[:, -1:]
printTensor(inputs, globals(), PRINT_TENSOR_OUTPUT)
out=decoder(inputs)
printTensor(out, globals(), PRINT_TENSOR_OUTPUT)
query=decoder.hidden.permute(1,0,2)
printTensor(query, globals(), PRINT_TENSOR_OUTPUT)

alphas=calc_alphas(keys, query)
printTensor(alphas, globals(), PRINT_TENSOR_OUTPUT)

context_vector=torch.bmm(alphas,values)
printTensor(context_vector, globals(), PRINT_TENSOR_OUTPUT)

concatenated=torch.cat([context_vector, query], axis=-1)
printTensor(concatenated, globals(), PRINT_TENSOR_OUTPUT)

products=torch.bmm(query, keys.permute(0,2,1))
printTensor(products, globals(), PRINT_TENSOR_OUTPUT)

alphas=F.softmax(products, dim=-1)
printTensor(alphas, globals(), PRINT_TENSOR_OUTPUT)


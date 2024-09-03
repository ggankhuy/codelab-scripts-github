# data format
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
from common.classes import *

from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep
from plots.chapter8 import plot_data
from plots.chapter9 import sequence_pred
import matplotlib.pyplot as plt

CONFIG_ALPHAS_SYNTHETIC_CALC=0

def calc_alphas(ks,q):
    printDbg("calc_alphas entered...")
    printTensor(ks, globals(),"brief")
    printTensor(q, globals(), "brief")
 
   if CONFIG_ALPHAS_SYNTHETIC_CALC:
        N,L,H=ks.size()
        print("ks.size(): ", ks.size())
        print("N,L,H: ", N,L,H)
        alphas=torch.ones(N,1,L).float()*1/L
    else:
        # N,1,H x N,H,L=>N,1,L
        # [batch, 1, hidden] x [batch, hidden, len] => [batch, 1, len]
        # [1, hidden] x [hidden, len] = [1, len]
        products = torch.bmm(q, ks.permute(0,2,1))
        alphas=F.softmax(products, dim=-1)

    printTensor(alphas, globals(), "brief")
    return alphas

full_seq=(torch.tensor([[-1,-1],[1,-1], [1,1],[1,-1]]).float().view(1,4,2))
source_seq = full_seq[:, :2]
target_seq = full_seq[:, 2:]
printTensor(full_seq, globals(), "full") 
printTensor(source_seq, globals(), "full") 
printTensor(target_seq, globals(), "full") 

torch.manual_seed(21)
encoder=Encoder(n_features=2, hidden_dim=2)
hidden_seq=encoder(source_seq)
values=hidden_seq
keys=hidden_seq
printTensor(values, globals(), "full")
#exprintTensor(keys, globals(), "full")

torch.manual_seed(21)
decoder=Decoder(n_features=2, hidden_dim=2)
decoder.init_hidden(hidden_seq)
inputs=source_seq[:, -1:]
printTensor(inputs, globals(), "full")
out=decoder(inputs)
printTensor(out, globals(), "full")
query=decoder.hidden.permute(1,0,2)
printTensor(query, globals(), "full")

alphas=calc_alphas(keys, query)
printTensor(alphas, globals(), "full")

context_vector=torch.bmm(alphas,values)
printTensor(context_vector, globals(), "full")

concatenated=torch.cat([context_vector, query], axis=-1)
printTensor(concatenated, globals(), "full")

products=torch.bmm(query, keys.permute(0,2,1))
printTensor(products, globals(), "full")

alphas=F.softmax(products, dim=-1)
printTensor(alphas, globals(), "full")


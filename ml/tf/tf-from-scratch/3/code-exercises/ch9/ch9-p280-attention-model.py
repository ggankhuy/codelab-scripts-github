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

full_seq=(torch.tensor([[-1,-1],[1,-1], [1,1],[1,-1]]).float().view(1,4,2))
source_seq = full_seq[:, :2]
target_seq = full_seq[:, 2:]
printTensor(full_seq, globals(), "full")
printTensor(source_seq, globals(), "full")
printTensor(target_seq, globals(), "full")

torch.manual_seed(21)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder_attn = DecoderAttn(n_features=2, hidden_dim=2)

# Generates hiddenstates (keys and values)

hidden_seq = encoder(source_seq)
decoder_attn.init_hidden(hidden_seq)

# target sequence  generation

inputs = source_seq[:, -1:]
target_len = 2

for i in range(target_len):
    out = decoder_attn(inputs)
    printTensor(out, globals(), "full")
    inputs = out

        

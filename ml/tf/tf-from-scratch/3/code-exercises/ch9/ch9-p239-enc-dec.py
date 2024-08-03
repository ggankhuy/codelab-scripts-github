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
import matplotlib.pyplot as plt

CONFIG_ENABLE_PLOT=0

if CONFIG_ENABLE_PLOT:
    fig = plot_data(points, directions, n_rows=1)
    plt.show()

full_seq = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2) # starting from LOWER LEFT ,CCW.

source_seq = full_seq[:, :2] # first two corners
target_seq = full_seq[:, 2:] # last two corners

torch.manual_seed(21)

encoder = Encoder(n_features=2, hidden_dim=2)
hidden_seq = encoder(source_seq) # output is N, L, F
printTensor(hidden_seq, globals(), "full")
hidden_final = hidden_seq[:, -1:]   # takes last hidden state
printTensor(hidden_final, globals(), "full")

decoder = Decoder(n_features=2, hidden_dim=2)
decoder.init_hidden(hidden_seq)
inputs = source_seq[:, -1:]

printTensor(inputs, globals())

target_len=2

for i in range(target_len):
    print("------ loop: ", i, "------")
    printTensor(decoder.hidden, getGlobalsClass(decoder), "full")
    out=decoder(inputs)
    inputs = out
    printTensor(out, globals(), "full")




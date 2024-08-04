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

full_seq = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]]).float().view(1, 4, 2) # starting from LOWER LEFT ,CCW.


CONFIG_ENABLE_TEACHER_ENFORCING_NONE=0 #241
CONFIG_ENABLE_TEACHER_ENFORCING_RANDOM=1 #241
CONFIG_ENABLE_TEACHER_ENFORCING_ALWAYS=2 #241
CONFIG_ENABLE_TEACHER_ENFORCING=CONFIG_ENABLE_TEACHER_ENFORCING_RANDOM

if CONFIG_ENABLE_PLOT:
    fig = plot_data(points, directions, n_rows=1)
    plt.show()


source_seq = full_seq[:, :2] # first two corners
target_seq = full_seq[:, 2:] # last two corners

torch.manual_seed(21)

# create encoder 

encoder = Encoder(n_features=2, hidden_dim=2)

# creaet decoder

decoder = Decoder(n_features=2, hidden_dim=2)

encdec=EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)
encdec.train()
outputs=encdec.forward(full_seq)
printTensor(outputs, globals(), "full")

encdec.eval()
outputs_from_src_seq=encdec(source_seq)
printTensor(outputs_from_src_seq, globals(), "full")


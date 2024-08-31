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

# N,1,H=batch, 1, hidden
q=torch.tensor([.55,.95]).view(1,1,2)
printTensor(q, globals(), "full")
k=torch.tensor([[0.65, 0.2],[0.85, -0.4],[-0.95, -0.75]]).view(1,3,2) # N,L,H=batch, length, hidden
printTensor(k, globals(), "full")

# N,1,H x N,H,L => N,1,L  = batch,1,hidden x batch,hidden,length => batch,1,len
prod=torch.bmm(q,k.permute(0,2,1))
printTensor(prod, globals(), "full")



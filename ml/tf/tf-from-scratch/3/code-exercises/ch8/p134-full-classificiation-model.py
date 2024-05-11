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
test_points, test_directions = generate_sequences(seed=19)
train_data = TensorDataset(torch.as_tensor(points).float(),torch.as_tensor(directions).view(-1,1).float())
test_data = TensorDataset(torch.as_tensor(test_points).float(), torch.as_tensor(test_directions).view(-1,1).float())
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data,  batch_size=16)

torch.manual_seed(21)
model=SquareModel(n_features=n_features, hidden_dim=hidden_dim, n_outputs=1)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


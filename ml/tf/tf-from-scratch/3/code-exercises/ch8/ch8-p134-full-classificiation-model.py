import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset,  random_split, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('..')

from common.settings import *
from common.classes import *
from data_generation.square_sequences import generate_sequences

from stepbystep.v4 import StepByStep
from stepbystep.v4 import *

from plots.chapter8 import plot_data

CONFIG_USE_SBS=1
CONFIG_ENABLE_PLOT_LOSS=1
CONFIG_ENABLE_PLOT_PRED=1
CONFIG_COMPUTE_ACCURACY=1

print("Import setings:")
printDbg("hidden_dim: ", hidden_dim)
printDbg("n_features: ", n_features)

CONFIG_PLOT=0
points, directions = generate_sequences(256,  seed=13)
test_points, test_directions = generate_sequences(seed=19)

printTensor(points, globals())
printTensor(test_points, globals())

train_data = TensorDataset(torch.as_tensor(points).float(),torch.as_tensor(directions).view(-1,1).float())
test_data = TensorDataset(torch.as_tensor(test_points).float(), torch.as_tensor(test_directions).view(-1,1).float())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data,  batch_size=16)

printTensor(train_loader, globals())
printTensor(test_loader, globals())

torch.manual_seed(21)
model=SquareModel(n_features=n_features, hidden_dim=hidden_dim, n_outputs=1)
loss  = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
writer = None
scheduler = None

if CONFIG_USE_SBS:
    sbs_rnn=StepByStep(model, loss, optimizer)
    sbs_rnn.set_loaders(train_loader, test_loader)
    sbs_rnn.train(100)
else:
    pass 

if CONFIG_ENABLE_PLOT_LOSS:
    fig=sbs_rnn.plot_losses()
    plt.show()

if CONFIG_COMPUTE_ACCURACY:
    state=model.basic_rnn.state_dict()
    print(state['weight_ih_l0'], state['bias_ih_l0'])

    output=StepByStep.loader_apply(test_loader, sbs_rnn.correct)
    printTensor(output, globals(), "full")

if CONFIG_ENABLE_PLOT_PRED:
    print(type(sbs_rnn.yhat_acc))
    print(numpy.array(sbs_rnn.yhat_acc).shape)
    print(numpy.array(sbs_rnn.yhat_acc)[0:10])


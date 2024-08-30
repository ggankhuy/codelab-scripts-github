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

def calc_alphas(ks,q):
    # N,1,H x N,H,L=>N,1,L
    # [batch, 1, hidden] x [batch, hidden, len] => [batch, 1, len]
    # [1, hidden] x [hidden, len] = [1, len]
    products = torch.bmm(q, ks,permute(0,2,1))
    alphas=F.softmax(dim=-1)

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

'''
points, directions = generate_sequences(256,  seed=13)
printTensor(points, globals(), "brief")
printTensor(directions, globals(), "brief")

full_train=torch.as_tensor(points).float()
target_train=full_train[:, 2:] # all rectangles, along with last two corners, along with coords

printTensor(full_train, globals(), "brief")
printTensor(target_train, globals(), "brief")

test_points, test_directions = generate_sequences(seed=19)

printTensor(test_points, globals())
printTensor(test_directions, globals())

full_test = torch.as_tensor(test_points).float()
source_test = full_test[:, :2]
target_test = full_test[:, 2:]

printTensor(source_test, globals())
printTensor(target_test, globals())

train_data = TensorDataset(full_train, target_train)
test_data = TensorDataset(source_test, target_test)

#printTensor(train_data, globals())

generator = torch.Generator()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, generator=generator)
test_loader = DataLoader(test_data, batch_size=16)

printTensor(train_loader, globals())
printTensor(test_loader, globals())

CONFIG_ENABLE_PLOT=1

CONFIG_ENABLE_TEACHER_ENFORCING_NONE=0 #241
CONFIG_ENABLE_TEACHER_ENFORCING_RANDOM=1 #241
CONFIG_ENABLE_TEACHER_ENFORCING_ALWAYS=2 #241
CONFIG_ENABLE_TEACHER_ENFORCING=CONFIG_ENABLE_TEACHER_ENFORCING_RANDOM

if CONFIG_ENABLE_PLOT:
    fig = plot_data(points, directions, n_rows=1)
    plt.show()

# create encoder 

torch.manual_seed(23)
encoder = Encoder(n_features=2, hidden_dim=2)
decoder = Decoder(n_features=2, hidden_dim=2)
model=EncoderDecoder(encoder, decoder, input_len=2, target_len=2, teacher_forcing_prob=0.5)
loss=nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

sbs_seq=StepByStep(model, loss, optimizer)
sbs_seq.set_loaders(train_loader, test_loader)
sbs_seq.train(100)

if CONFIG_ENABLE_PLOT:
    fig=sbs_seq.plot_losses()
    plt.show()

    fig=sequence_pred(sbs_seq, full_test, test_directions)
    plt.show()

'''

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

# q=decoder hidden state:       N,L,1: H=batch=1, len=1, H=hidden=2

q=torch.tensor([.55,.95]).view(1,1,2)
printTensor(q, globals(), "full")

# k=encoder hidden state:       N,L,H: N=batch=1, L=length=3, H=hidden=2.

k=torch.tensor([[0.65, 0.2],[0.85, -0.4],[-0.95, -0.75]]).view(1,3,2)
printTensor(k, globals(), "full")

# N,1,H x N,L,H.permute(0,2,1) => N,L,H x N,H,L => [n=1, len=1, h=2] x [n=1, h=2, l=3] => [n=1, l(q)=1, l=3(k)]
# review matrix multiplication sentimental explanation: https://mkang32.github.io/python/2020/08/23/dot-product.html
# [LEN=1, H=2] X [H=2, L=3] => L=1(Q), L=3(K) => [1,2][2,3]=[1,3]  => 
# [.55 0.96] x [0.65 0.85 -0.95]
#              [0.2 -0.4   -0.75] => [0.55*0.65+0.96*0.2 0.55*0.85+0.96*-0.4 0.55*-0.95+0.96*-0.75] = [0.3575+0.192=0.5495 0.4675+-0.384=0.0835 -0.5225+-0.72=-1.24]=
#                                   = [ 0.5495 0.0835 -1.24]

prod=torch.bmm(q,k.permute(0,2,1))
printTensor(prod, globals(), "full")

scores=F.softmax(prod, dim=-1)
printTensor(scores,globals(), "full")
print("sum(scores): ", torch.sum(scores))

v=k

# context=\
# torch.bmm(scores,                                             v)=\
# torch.bmm(F.softmax(prod,                         dim=-1),    v)=\
# torch.bmm(F.softmax(torch.bmm(q,k.permute(0,2,1), dim=-1),    v)

context=torch.bmm(scores,v)
printTensor(context, globals(), "full")


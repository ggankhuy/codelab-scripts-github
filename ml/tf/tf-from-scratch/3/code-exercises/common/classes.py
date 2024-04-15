import copy
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('.')
from common.settings import *

class RNNCell:
    def __init__(self, rnn_cell_src: nn.RNNCell, input_size:int, hidden_size:int):
        self.linear_input=nn.Linear(input_size, hidden_size)
        self.linear_hidden=nn.Linear(hidden_size, hidden_size)
        rnn_state=rnn_cell_src.state_dict()

        with torch.no_grad():
            self.linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
            self.linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
            self.linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
            self.linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

        self.initial_hidden=torch.zeros(1, hidden_size)
        printDbg(self.initial_hidden)
        self.th=self.linear_hidden(self.initial_hidden)
        printDbg(self.th)
        self.tx=0
        self.th=0
        self.final_hidden=0

    def forward(self, x):
        self.tx=self.linear_input(x)
        adding=self.th+self.tx
        printDbg(adding)
        self.final_hidden=torch.tanh(adding)


import copy
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('.')
from common.settings import *

def log_methods(cls):
    for name, value in vars(cls).items():
        if callable(value):
            setattr(cls, name, log_method(value))

def log_method(func):
    def wrapper(*args, **kwargs):
        print(f"Calling method: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

#@log_methods
class Linear:
    def printFcn(func):
        def inner(func, *argv):
            print(func, type(func))
        return inner

    @printFnc
    def __init__(self, input_size:int, hidden_size:int):
        self.input_size=input_size
        self.ihddn_dim=hidden_size
        self.weight=torch.zeros([input_size, hidden_size], requires_grad=True)
        self.bias=torch.zeros(hidden_size)
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return  torch.matmul(X, self.weight.T) + self.bias

#@log_methods
class RNNCell:
    def __init__(self, rnn_cell_src: nn.RNNCell, input_size:int, hidden_size:int):
        printDbg("RNNCell.__init__ entered: input_size, hidden_size: ", input_size, hidden_size)
        rnn_state_src=rnn_cell_src.state_dict()
        printDbg("rnn_state_src: \n", rnn_state_src)
        self.linear_input=Linear(input_size, hidden_size)
        self.linear_hidden=Linear(hidden_size, hidden_size)

        with torch.no_grad():
            self.linear_input.weight=nn.Parameter(rnn_state_src['weight_ih'])
            self.linear_input.bias=nn.Parameter(rnn_state_src['bias_ih'])
            self.linear_hidden.weight=nn.Parameter(rnn_state_src['weight_hh'])
            self.linear_hidden.bias=nn.Parameter(rnn_state_src['bias_hh'])

        # hidden is for rnn, remember! we are inside rnn class not linear class.

        self.initial_hidden=torch.zeros(1, hidden_size)
        self.th=self.linear_hidden(self.initial_hidden)
        printDbg("RNNCell.__init__: self.th computed to: ", self.th)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        printDbg("RNNCell.forward entered(x=" + str(x))
        self.tx=self.linear_input(x)
        printDbg("RNNCell.forward: self.tx computed to: ", self.tx)
        adding=self.th+self.tx
        printDbg(adding)
        self.final_hidden=torch.tanh(adding)
        printDbg("RNNCell.forward: returning self.final_hidden: ", self.final_hidden)
        return self.final_hidden

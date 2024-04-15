import copy
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('.')
from common.settings import *
CONFIG_USE_NN_LINEAR=0

class Linear:
    def printFcn(func):
        def inner(func, *argv):
            print(func.__name__)
        return inner

    def __init__(self, input_size:int, hidden_size:int):
        self.input_size=input_size
        self.ihddn_dim=hidden_size
        self.weight=torch.zeros([input_size, hidden_size], requires_grad=True)
        self.bias=torch.zeros(hidden_size)
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        printDbg("Linear.forward entered(X=" + str(X))
        out = torch.matmul(X, self.weight)
        out += self.bias

        '''
        if not out:
            print("Error: out is None!!!")
        '''
        printDbg("returning out: \n", out)
        return out
        

class RNNCell:
    def __init__(self, rnn_cell_src: nn.RNNCell, input_size:int, hidden_size:int):
        printDbg("RNNCell.__init__ entered: input_size, hidden_size: ", input_size, hidden_size)

        if CONFIG_USE_NN_LINEAR: 
            self.tx=0
            self.th=0
            self.final_hidden=0
            self.linear_input=nn.Linear(input_size, hidden_size)
            self.linear_hidden=nn.Linear(hidden_size, hidden_size)
            rnn_state=rnn_cell_src.state_dict()

            with torch.no_grad():
                self.linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
                self.linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
                self.linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
                self.linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

            self.initial_hidden=torch.zeros(1, hidden_size)
            printDbg("self.initial_hidden: ", self.initial_hidden)
            self.th=self.linear_hidden(self.initial_hidden)
            printDbg("self.th", self.th)
        else:
            rnn_state=rnn_cell_src.state_dict()

            self.linear_input=Linear(input_size, hidden_size)
            self.linear_hidden=Linear(hidden_size, hidden_size)

            with torch.no_grad():
                self.linear_input.weight=nn.Parameter(rnn_state['weight_ih'])
                self.linear_input.bias=nn.Parameter(rnn_state['bias_ih'])
                self.linear_hidden.weight=nn.Parameter(rnn_state['weight_hh'])
                self.linear_hidden.bias=nn.Parameter(rnn_state['bias_hh'])

            # hidden is for rnn, remember! we are inside rnn class not linear class.

            self.initial_hidden=torch.zeros(1, hidden_size)
            printDbg("self.initial_hidden: ", self.initial_hidden)
            self.th=self.linear_hidden(self.initial_hidden)
            printDbg("self.th", self.th)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        printDbg("RNNCell.forward entered(x=" + str(x))
        self.tx=self.linear_input(x)
        printDbg("RNNCell.foward: self.th: ", self.th)
        adding=self.th+self.tx
        printDbg(adding)
        self.final_hidden=torch.tanh(adding)
        printDbg("RNNCell.forward: returning self.final_hidden: ", self.final_hidden)
        return self.final_hidden

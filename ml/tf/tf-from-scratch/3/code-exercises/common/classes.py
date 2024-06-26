import copy
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('.')
from common.settings import *

class Linear:
#   @printFnc
    def __init__(self, input_size:int, hidden_size:int):
        self.input_size=input_size
        self.ihddn_dim=hidden_size
        self.weight=torch.zeros([input_size, hidden_size], requires_grad=True)
        self.bias=torch.zeros(hidden_size)
#   @printFnc
    def __call__(self, X):
        return self.forward(X)

#   @printFnc
    def forward(self, X):
        return  torch.matmul(X, self.weight.T) + self.bias

debug_rnn_cell=0
def printDbgRnnCell(*argv):
    if debug_rnn_cell:
        printDbg(argv)

class RNNCell:
    def __init__(self, rnn_cell_src: nn.RNNCell, input_size:int, hidden_size:int):
        printDbgRnnCell("RNNCell.__init__ entered: input_size, hidden_size: ", input_size, hidden_size)
        rnn_state_src=rnn_cell_src.state_dict()
        printDbgRnnCell("rnn_state_src: \n", rnn_state_src)
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
        printDbgRnnCell("RNNCell.__init__: self.th computed to: ", self.th)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        printDbgRnnCell("RNNCell.forward entered(x=" + str(x))
        self.tx=self.linear_input(x)
        printDbgRnnCell("RNNCell.forward: self.tx computed to: ", self.tx)
        adding=self.th+self.tx
        printDbgRnnCell(adding)
        self.final_hidden=torch.tanh(adding)
        printDbgRnnCell("RNNCell.forward: returning self.final_hidden: ", self.final_hidden)
        return self.final_hidden

class SquareModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs):
        super(SquareModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = None

        # Simple RNN

        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first = True)
    
        # classifier to produce as many logits as outputs

        self.classifiers = nn.Linear(self.hidden_dim, self.n_outputs)

    def forward(self, X):
        # X is batch first (N,L,F)
        # output is (N,L,H)
        # final hidden state is (1,N,H)

        
        batch_first_output, self.hidden = self.basic_rnn(X)
        
        # only last item in sequence (N,1,H)

        last_output = batch_first_output[:, -1]
         
        # classifier will output (N,1,n_outputs)

        out = self.classifier(last_output)

        # final outputs is (N, n_outputs)
        return out.view(-1, self.n_outputs)
        


 



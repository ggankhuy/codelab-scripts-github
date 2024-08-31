import copy
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append('.')
from common.settings import *

def printModelInfo(pModel, pModelName=None):
    if pModelName:
        printDbg("model info for ", pModelName, ": ")
    model_state=pModel.state_dict()
    for k, v in model_state.items():
        printDbg(k, np.array(v).shape, "\n", v)
    return model_state

class Linear:
    @printFnc
    def __init__(self, input_size:int, hidden_size:int):
        self.input_size=input_size
        self.hidden_dim=hidden_size
        self.weight=torch.zeros([input_size, hidden_size], requires_grad=True)
        self.bias=torch.zeros(hidden_size)

        # this is not necessary but only for plotting.

        self.hidden=torch.zeros(hidden_size)
#   @printFnc
    def __call__(self, X):
        return self.forward(X)

#   @printFnc
    def forward(self, X):
        self.hidden = torch.matmul(X, self.weight.T) + self.bias
        return self.hidden

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
        self.adding=0

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
        self.tx = self.linear_input(x)

        # this is already done in init().
        #self.th = self.linear_hidden(hidden)

        printDbgRnnCell("RNNCell.forward: self.tx computed to: ", self.tx)
        self.adding = self.th+self.tx
        printDbgRnnCell(self.adding)
        self.final_hidden = torch.tanh(self.adding)
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

        self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)

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
        
class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        print("Encoder.init(n_features =", n_features,", hidden_dim =", hidden_dim, ")")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn =  nn.GRU(self.hidden_dim, self.n_features, batch_first = True)
        
    def forward(self, X):
        print("Encoder.forward(X=",X.shape,")")
        rnn_out, self.hidden = self.basic_rnn(X)
        return rnn_out # N, L, F.

class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.hidden = None
        self.basic_rnn = nn.GRU(self.n_features, self.hidden_dim, batch_first=True) 
        self.regression = nn.Linear(self.hidden_dim, self.n_features)
        
    def init_hidden(self, hidden_seq):

        # We only need the final hidden state

        hidden_final = hidden_seq[:, -1:] # N, 1, H = length, last of batch, features [1,1,2]

        # But we need to make it sequence-first

        self.hidden = hidden_final.permute(1, 0, 2) # 1, N, H = last of barch, length, features=> [1,4,2]
        
    def forward(self, X):

        # X is N, 1, F = [1,1,2] in example

        batch_first_output, self.hidden = self.basic_rnn(X, self.hidden) 
        last_output = batch_first_output[:, -1:]
        out = self.regression(last_output)
        
        # N, 1, F

        return out.view(-1, 1, self.n_features) 

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, input_len, target_len, teacher_forcing_prob=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_len = input_len
        self.target_len = target_len
        self.teacher_forcing_prob = teacher_forcing_prob
        self.outputs = None

    def init_outputs(self, batch_size):
        device = next(self.parameters()).device
        
        # N, L (target), F

        self.outputs= torch.zeros(batch_size, self.target_len, self.encoder.n_features).to(device)

    def store_output(self, i, out):
        # stores the output.

        self.outputs[:, i:i+1,:] = out

    def forward(self, X):

        # splits the data into source and target sequences
        # target seq will be empty in testing mode.

        source_seq = X[:, :self.input_len, :]
        target_seq = X[:, self.input_len:, :]
        self.init_outputs(X.shape[0])

        #Encode expects N,L,F

        hidden_seq = self.encoder(source_seq)
        
        #Output is N,L,H

        self.decoder.init_hidden(hidden_seq)

        # Last inputs of the encoder is also first input of decoder

        dec_inputs = source_seq[:, -1:, :]

        # Generates many of outputs as target_len

        for i in range(0, self.target_len):
            # output of decoder is 1, L, F

            out = self.decoder(dec_inputs)
            self.store_output(i, out)
 
        prob = self.teacher_forcing_prob

        # In evaluation / test the target sequence is unknown, so 
        # we can not use teacher enforcing if not self.training

        if not self.training:
            prob = 0

        # if it is teacher forcing 

        if torch.rand(1) <= prob:
            # takes the actual element

            dec_inputs = target_seq[:, i:i+1, :]
        else:
            # otherwise uses last predicte output

            dec_inputs = out

        return self.outputs


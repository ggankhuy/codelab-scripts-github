import torch
import torch.nn as nn
torch.manual_seed(1)

rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)
w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

# w_xh: [2,5] - input hidden layer weight.
# x_hh: [2,2] - recurrent hidden layer weight.
# b_xh: [2] - input hidden layer bias
# b_hh: [2] - recurrent hidden layer bias.

print("W_xh shape: ", w_xh.shape)
print("W_hh shape: ", w_hh.shape)
print("B_xh shape: ", b_xh.shape)
print("B_hh shape: ", b_hh.shape)
    
# x_seq: 3,5
# 
# matmul input layer: ( x_seq, w_xh) = [3,5] x [5,2] = [3,2]
# matmul hidden layer: (output of input layer, x_hh) = [3,2] x [2,2] = [3,2]

x_seq = torch.tensor([[1.0]* 5, [2.0]*5, [3.0]*5]).float()
print(f'x_seq: {x_seq}, shape: {x_seq.shape}')

## output  of the simple RNN:
# after torch reshape of x_seq: [3,5] to [1,3,5]
#1st dim: batch size/count?
#2nd dim: sequence
#3rd dim: features

output, hn = rnn_layer(torch.reshape(x_seq, (1,3,5)))

## manually computing the output:

out_man = []

## Run a time series run...

for t in range(3):

    # reshape x_seq[t] from [5] to [1,5]
    # x_seq[t] extracts first batch as 1st dim is batch count.

    # reshape output: [5] -> [1,5]
    
    xt = torch.reshape(x_seq[t], (1,5))
    print(f'Time step {t} => ')
    print('     Input       :', xt.numpy())


    # 1A. INPUT HIDDEN LAYER OUTPUT COMPUTATION.
    # input * weight.input + bias.input_layer
    # [1,5] * [5,2] + [2] => [1,2] 

    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
    print('     Hidden      :', ht.detach().numpy())

    # store previous output (time or sequence)
    # prev_h: [1,2]

    if t > 0:
        prev_h = out_man[t-1]
    else:
        prev_h = torch.zeros((ht.shape))

    # 1B. RECURRENT HIDDEN LAYER COMPUTATION. Note that ot=ht+w_hh*prev_h=w_xh * xt +  w_hh * prev_h.
    # output = output.previous or zero init * weight.hidden + bias.hidden_layer
    # [1,2] * [2,2] + [2] => [1,2] 

    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh,0,1)) + b_hh

    # update output, which will be assigned to prev_h in next iteration.

    ot = torch.tanh(ot)
    out_man.append(ot)

    print('     Output (manual)     :', ot.detach().numpy())
    print('     RNN output          :', output[:, t].detach().numpy())
    print()


# at the end out_man is list with [3], with each of them [1,2] -> [3,2]

# need to study more about how followin works, specially parameters
# torch.squeeze - redce 
# torch.transpose

import torch
import numpy as np
np.set_printoptions(precision=3)

cuda = torch.device('cuda')

# creating tensor from array, np.array

a=[1,2,3]
b=np.array([4,5,6], dtype=np.int32)
t_a=torch.tensor(a, device=cuda)
t_b=torch.from_numpy(b)
print(t_a)
print(t_b)

# creating 1-filled tensor

t_ones=torch.ones(2,3)
print(t_ones)
print(t_ones.shape)

# creating random value tensor

t_random=torch.rand(2,3)
print(t_random)
print(t_random.shape)

# converting to a newtype

t_a_new=t_a.to(torch.int64)
print(t_a_new)
print(t_a_new.dtype)

# transposing a tensor

t=torch.rand(3,5)
t_tr=torch.transpose(t, 0, 1)
print("t_tr: ", t.shape, "-->", t_tr.shape)

# reshape, 1d to 2d

t=torch.zeros(30)
t_reshape=t.reshape(30, 1)
print("t_reshape: ", t_reshape, "\n", t_reshape.shape)

# removing unnecessary dimensions.

t=torch.zeros(1,2,1,4,1)
t_sqz=torch.squeeze(t, 2)
print("t: ", t)
print("t: ", t_sqz)
print(t.shape, "->", t_sqz.shape)

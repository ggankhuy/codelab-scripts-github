import torch
import numpy as np

#split tensors
torch.manual_seed(1)
t=torch.rand(6,3)
print(t)
t_splits=torch.chunk(t, 3, 0)
print(t_splits)
t_splits=torch.chunk(t, 3, 1)
print(t_splits)

#concat tensors
A = torch.ones(3, 5)
B = torch.zeros(2, 5)
A1 = torch.ones(3, 5)
B1 = torch.zeros(3, 4)
C = torch.cat([A, B], axis=0)
print(C)
C = torch.cat([A1, B1], axis=1)
print(C)

#STACK
A=torch.ones(3)
B=torch.zeros(3)
S=torch.stack([A,B], axis=1)
print(A)
print(B)
print(S)

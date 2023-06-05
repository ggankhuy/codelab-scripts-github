import torch
import numpy as np
np.set_printoptions(precision=3)

torch.manual_seed(1)
t1=2*torch.rand(5,2)-1
t2=torch.rand(5,2)-1
print(t1)

# not working!
#t2=torch.normal(means=0, std=1, size=(5,2))
print(t2)

#element wise multiplication, not dot or cross product.

t3=torch.multiply(t1,t2)
print(t3)

t4=torch.mean(t1, axis=1)
print(t4)

#MM
t5=torch.matmul(t1, torch.transpose(t2,0,1))
print(t5)
print(t1.shape, torch.transpose(t2,0,1).shape, "->", t5.shape)

#MM
t6=torch.matmul(torch.transpose(t1,0,1), t2)
print(t6)
print(torch.transpose(t2,0,1).shape, t1.shape, "->", t6.shape)

# norm

norm_t1=torch.linalg.norm(t1, ord=2, dim=1)
print(norm_t1)

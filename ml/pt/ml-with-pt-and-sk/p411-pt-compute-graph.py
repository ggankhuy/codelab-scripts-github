import torch
import torch.nn as nn

cuda = torch.device('cuda')

def compute_z(a, b, c):
    r1=torch.sub(a, b)
    r2=torch.mul(r1, 2)
    z=torch.add(r2, c)
    return z

a=torch.tensor(3.14, requires_grad=True, device='cuda')
print(a)
b=torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device='cuda')
print(b)

torch.manual_seed(1)
w=torch.empty(2, 3)
nn.init.xavier_normal_(w)
print(w)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)

'''
Looking at the .code output, we can see that the if-else branch is nowhere to be found! Why? Tracing does exactly what we said it would: run the code, record the operations that happen and construct a ScriptModule that does exactly that. Unfortunately, things like control flow are erased.

How can we faithfully represent this module in TorchScript? We provide a script compiler, which does direct analysis of your Python source code to transform it into TorchScript. Let’s convert MyDecisionGate using the script compiler:
'''
import torch

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        return new_h, new_h

x = torch.rand(3, 4)
h = torch.rand(3, 4)

my_cell = MyCell(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)

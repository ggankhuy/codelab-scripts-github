import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

x = torch.rand(3, 4)
h = torch.rand(3, 4)

my_cell = MyCell()
print(my_cell)
print(my_cell(x, h))

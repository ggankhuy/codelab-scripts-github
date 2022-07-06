import torch

#w=torch.tensor(1.0, requires_grad=True)
#b=torch.tensor(0.5, requires_grad=True)

TENSOR_SIZE=10
w=torch.rand(TENSOR_SIZE, requires_grad=True)
b=torch.rand(TENSOR_SIZE, requires_grad=True)

torch.manual_seed(1)
x=torch.rand(TENSOR_SIZE)
y=torch.rand(TENSOR_SIZE)

#x=torch.tensor([1.4])
#y=torch.tensor([2.1])
z=torch.add(torch.mul(w, x), b)
loss = (y-z).pow(2).sum()
loss.backward()
print("w: ", w)
print("b: ", b)
print('dL/dw : ', w.grad)
print('dL/db : ', b.grad)

# verifying output of loss.backward...

test1=2 * x * ((w*x+b)-y)
print(test1[:5])


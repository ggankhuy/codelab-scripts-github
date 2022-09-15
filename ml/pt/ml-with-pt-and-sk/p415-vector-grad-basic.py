import torch
import code
cuda = torch.device('cuda')

# Create weight and bias values.

TENSOR_SIZE=5
w=torch.rand(TENSOR_SIZE, requires_grad=True, device='cuda')
b=torch.rand(TENSOR_SIZE, requires_grad=True, device='cuda')

torch.manual_seed(1)

# Create input(x), output (y, expected).
# input(x) used for forward pass: z=w*x+b, z will be computed y rather than expected y. diff=(z-y)

x=torch.rand(TENSOR_SIZE, device='cuda')
y=torch.rand(TENSOR_SIZE, device='cuda')

for i in range(0, 5):
    print("-------- ", i, " ---------")
    z=torch.add(torch.mul(w, x), b)
    loss = (y-z).pow(2).sum()

    loss.backward()
    print("loss: ", loss)
    print("w: ", w, type(w))
    print("b: ", b, type(b))
    print('dL/dw : ', w.grad, type(w.grad))
    print('dL/db : ', b.grad, type(b.grad))

    # verifying output of loss.backward...

    print("verifying output of loss.backward...(compare with DL/DW)")
    test1=2 * x * ((w*x+b)-y)
    print("dL/dw    : ", w.grad)
    print("t        : ", test1[:5])

    # update weights

    w1 = w + w.grad
    b1 = b + b.grad
    w=w1.detach()
    w.requires_grad=True
    b=b1.detach()
    b.requires_grad=True

    print("new updated w1/b1: ")
    print("w: ", w, type(w))
    print("b: ", b, type(b))


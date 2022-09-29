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


    # test1=DL/Dw = DL/DZ * DZ/DW 
    # 1. DL/DZ=D/DZ (y-z)**2 = D/DZ y**2-2yz+z**2 = -2y + 2z = 2(z-y)
    # 2. DZ/DW = D/DW w * x + b = x.
    # 3. DL/DW = DL/DZ * DZ/DW = 2(z-y)x = 2x(z-y)  = 2x(w*x+B)-y

    # test2=DL/db = DL/DZ * DZ/Db
    # 1a. same as 1.
    # 2a.DW/DB = d/db w * x  + b = 1
    # 3a. DL/Db = DL/DZ * DZ/Db = 2(z-y) * 1 = 2(z-y) = 2 (w * x + b) -y
    
    test1=2 * x * ((w*x+b)-y)
    print("dL/dw    : ", w.grad)
    print("t        : ", test1[:5])

    test2=2 * ((w*x + b) - y)
    print("dL/db    : ", b.grad)
    print("t        : ", test2[:5])

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


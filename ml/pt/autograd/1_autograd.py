# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
# Section title: Differentiation in Autograd

import torch

# Take a look at how autograd collects gradients. We create two tensors a and b with requires_grad=True. 
# This signals to autograd that every operation on them should be tracked.

a = torch.tensor([2, 3, 12], dtype=torch.float32, requires_grad=True)
b = torch.tensor([6, 4, 5], dtype=torch.float32, requires_grad=True)

print("a: ", a.size())
print("b: ", b.size())
print("a/a.grad/a.grad.size(): ", a, "/", a.grad)
print("b/b.grad/b.grad.size(): ", b, "/", b.grad)

# We create another tensor Q from a and b.

Q = 3*a**3 - b**2

# Let's assume a and b to be parameters of an NN, and Q to be the error. 
# In NN training, we want gradients of the error w.r.t. parameters, i.e.

# dQ/da = 9a^2, dQ/db = -2b.

# When we call .backward() on Q, autograd calculates these gradients and 
# stores them in the respective tensor's .grad attribute.

# We need to explicitly pass a gradient argument in Q.backward() because it is a vector. 
# gradient is a tensor of the same shape as Q, and it represents the gradient of Q w.r.t. itself, i.e.
# dQ/dQ=1.

print("Q: ", Q.size(), Q)

# Equivalently, we can also aggregate Q into a scalar and call backward implicitly, like Q.sum().backward().

external_grad = torch.tensor([1, 1, 1], dtype=torch.float32)
print("external_grad: ", external_grad.size())
print("external_grad before backward(): ", external_grad, external_grad.grad)
Q.backward(gradient=external_grad)
print("external_grad after  backward(): ", external_grad, external_grad.grad)

# Gradients are now deposited in a.grad and b.grad and compared to dq/da, dq/db

print(9*a**2 == a.grad)
print(-2*b == b.grad)
print("a/a.grad/a.grad.size(): ", a, "/", a.grad)
print("b/b.grad/b.grad.size(): ", b, "/", b.grad)

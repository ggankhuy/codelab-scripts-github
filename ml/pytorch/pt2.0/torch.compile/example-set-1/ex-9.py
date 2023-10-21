import torch
import torch._dynamo

torch._dynamo.reset()

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 2:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)

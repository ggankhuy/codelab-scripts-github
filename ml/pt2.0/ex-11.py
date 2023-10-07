import torch

def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

def f2(x, y):
    return x + y

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)


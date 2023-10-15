import torch 

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

# Test that `fn1` and `fn2` return the same result, given
# the same arguments `args`. Typically, `fn1` will be an eager function
# while `fn2` will be a compiled function (torch.compile, TorchScript, or FX graph).
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)
print(inp1, "\n", inp2)

import torch
import traceback as tb

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

'''
traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))
'''

fx_f1 = torch.fx.symbolic_trace(f1, concrete_args={"x": inp1})
print("fx 1, 1:", test_fns(f1, fx_f1, (inp1, inp2)))
print("fx 1, 2:", test_fns(f1, fx_f1, (-inp1, inp2)))
'''
causes crash but still different than the one in torch ompile tutorial

try:
    torch.fx.symbolic_trace(f1)
except:
    tb.print_exc()
'''

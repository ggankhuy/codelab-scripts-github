import torch
import scipy 
import traceback as tb

def f3(x):
    x = x * 2
    x = scipy.fft.dct(x.numpy())
    x = torch.from_numpy(x)
    x = x * 2
    return x

'''
def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)
'''
inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

try:
    torch.jit.script(f3)
except:
    tb.print_exc()

try:
    torch.fx.symbolic_trace(f3)
except:
    tb.print_exc()

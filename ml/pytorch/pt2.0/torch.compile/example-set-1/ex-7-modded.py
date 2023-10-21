#  dont get confused by the fact thinking that test_fns(f1,traced_f1, (-inp1, inp2)
#  - test_fns is not comparing inp1 vs inp2
# it is comparing output of f1 and traced_f1 which "supposedly" return same value 
# (resulting in  allclose() being tru all the time), regardless of +/- of inp1, inp2.

# Moral of the story: no matter what are polarity of inp1, inp2, that either:
# ++, -+, +-, --, f1 and traced_f1 should always return true if worked correctly.

DEBUG=0 

import torch

def printDbg(arg, arg1=None, arg2=None, arg3=None): 
    if DEBUG:
        print(arg, arg1, arg2, arg3)

def f1(x, y):
    
    printDbg("f1 entered...")
    if x.sum() < 0:
        printDbg("returning -y")
        return -y
    printDbg("returning +y")
    return y

# Test that `fn1` and `fn2` return the same result, given
# the same arguments `args`. Typically, `fn1` will be an eager function
# while `fn2` will be a compiled function (torch.compile, TorchScript, or FX graph).
def test_fns(fn1, fn2, args):
    printDbg("\ntest_fns: entered, args: \n", args[0][0], "\n", args[1][0])
    out1 = fn1(*args)
    printDbg("  out1: ", out1[0])
    out2 = fn2(*args)
    printDbg("  out2: ", out1[0])
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

printDbg("created inp1, inp2: ")
printDbg(inp1[0])
printDbg(inp2[0])

traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 2:", test_fns(f1, traced_f1, (-inp1, inp2)))

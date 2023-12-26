import torch
import time

@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b
print(opt_foo2(torch.randn(10, 10), torch.randn(10, 10)))

for i in range(0,5):
    t1_foo = time.time_ns()
    foo(torch.randn(10, 10), torch.randn(10, 10))
    t2_foo = time.time_ns()
    print("time: unoptimized:   ", t2_foo-t1_foo)

for i in range(0,5):
    t1_foo1 = time.time_ns()
    opt_foo2(torch.randn(10, 10), torch.randn(10, 10))
    t2_foo1 = time.time_ns()
    print("time: optimized:     ", t2_foo1-t1_foo1)

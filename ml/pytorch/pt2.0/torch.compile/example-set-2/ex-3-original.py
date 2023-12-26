import torch


'''
Letânow demonstrate that using torch.compile can speed up real models. We will 
compare standard eager mode and torch.compile by evaluating and training a 
torchvision model on random data.


Notice that torch.compile takes a lot longer to complete compared to eager. 
This is because torch.compile compiles the model into optimized kernels as it 
executes. In our example, the structure of the model doesnâ€™t change, and so 
recompilation is not needed. So if we run our optimized model several more times, 
we should see a significant improvement compared to eager.


'''

# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

# Generates random input and targets data for the model, where `b` is
# batch size.
def generate_data(b):
    return (
        torch.randn(b, 3, 128, 128).to(torch.float32).cuda(),
        torch.randint(1000, (b,)).cuda(),
    )

N_ITERS = 10

from torchvision.models import densenet121
def init_model():
    return densenet121().to(torch.float32).cuda()

model = init_model()

# Reset since we are using a different mode.
import torch._dynamo
torch._dynamo.reset()

model_opt = torch.compile(model, mode="reduce-overhead")

inp = generate_data(16)[0]

with torch.no_grad():
    print("eager:", timed(lambda: model(inp))[1])
    print("compile:", timed(lambda: model_opt(inp))[1])

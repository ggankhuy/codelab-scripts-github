import torch

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
opt = torch.optim.Adam(model.parameters())

def train(mod, data):
    opt.zero_grad(True)
    pred = mod(data[0])
    loss = torch.nn.CrossEntropyLoss()(pred, data[1])
    loss.backward()
    opt.step()

eager_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, eager_time = timed(lambda: train(model, inp))
    eager_times.append(eager_time)
    print(f"eager train time {i}: {eager_time}")
print("~" * 10)

model = init_model()
opt = torch.optim.Adam(model.parameters())
train_opt = torch.compile(train, mode="reduce-overhead")

compile_times = []
for i in range(N_ITERS):
    inp = generate_data(16)
    _, compile_time = timed(lambda: train_opt(model, inp))
    compile_times.append(compile_time)
    print(f"compile train time {i}: {compile_time}")
print("~" * 10)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)




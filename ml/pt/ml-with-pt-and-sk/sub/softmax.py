import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import code
import numpy as np
#import PyQt5
from functools import wraps 
from datetime import datetime

m = nn.Softmax(dim=1)
#input = torch.randn(2, 3)
for i in [10, 20, 100]:
    print("----------------")
    input = torch.randn(2, i)
    print("input (softmax), sum:: ", input, "\n", input.sum())
    output = m(input)
    print("output (softmax), sum: ", output, "\n", output.sum())

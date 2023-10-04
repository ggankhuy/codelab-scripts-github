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

m = nn.Softmax2d()
#input = torch.randn(2, 3)
print("----------------")
input = torch.randn(1, 2, 2)
print("input (softmax), sum:: ", input, "\n", input.sum())
output = m(input)
print("output (softmax), sum: ", output, "\n", output.sum())

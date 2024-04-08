import copy 
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,  random_split, TensorDataset

import sys
sys.path.append('..')
from data_generation.square_sequences import generate_sequences
from stepbystep.v4 import StepByStep

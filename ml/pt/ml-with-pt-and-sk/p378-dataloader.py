import torch
import numpy as np
from torch.utils.data import DataLoader

t=torch.arange(6, dtype=torch.float32)
print(t)

data_loader=DataLoader(t)
for i in data_loader:
    print(i)


data_loader=DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)
    

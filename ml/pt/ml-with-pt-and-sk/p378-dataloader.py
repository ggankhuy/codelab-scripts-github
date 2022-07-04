import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

t=torch.arange(6, dtype=torch.float32)
print("t:", t)

data_loader=DataLoader(t)
for i in data_loader:
    print(i, ", ", i.dtype)


data_loader=DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)
    
for  batch in data_loader:
    print(f'batch: ', batch)

torch.manual_seed(1)
t_x = torch.rand([4,3], dtype=torch.float32)
print("t_x: ", t_x)
t_y = torch.arange(4)
print("t_y: ", t_y)


class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

joint_dataset = JointDataset(t_x, t_y)
for example in joint_dataset:
    print(' x: ', example[0], ' y: ', example[1])


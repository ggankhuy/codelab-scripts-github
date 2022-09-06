import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import time 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
ENABLE_PLOT=0
gpu = torch.device('cuda')

torch.manual_seed(1)
np.random.seed(1)
x=np.random.uniform(low=-1, high=1, size=(200, 2))
y=np.ones(len(x))

y[x[:,0] * x[:, 1]<0]=0
n_train=100

'''
x_train=torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train=torch.tensor(y[:n_train], dtype=torch.float32)
x_valid=torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid=torch.tensor(y[n_train:], dtype=torch.float32)

'''
x_train=torch.tensor(x[:n_train, :], dtype=torch.float32, device='cuda')
y_train=torch.tensor(y[:n_train], dtype=torch.float32, device='cuda')
x_valid=torch.tensor(x[n_train:, :], dtype=torch.float32, device='cuda')
y_valid=torch.tensor(y[n_train:], dtype=torch.float32, device='cuda')

fig=plt.figure(figsize=(6,6))

print(x[y==0, 0])
plt.plot(x[y==0, 0], x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0], x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)

if ENABLE_PLOT:
    plt.show()

model=nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
model.to('cuda')
print(model)

loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)

train_ds=TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
print("train_dl type: ", type(train_dl))
#time.sleep(3)

torch.manual_seed(1)
num_epochs=200

def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
#           print("x_batch/device, y_batch type/device: ", type(x_batch), x_batch.device, type(y_batch), y_batch.device)
#           time.sleep(3)
            pred=model(x_batch)[:, 0]
            loss=loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()

        loss_hist_train[epoch] /= n_train
        accuracy_hist_train[epoch] /= n_train/batch_size
        pred=model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred>=0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

history=train(model, num_epochs, train_dl, x_valid, y_valid)
for i in history:
    print("len: ", len(i))
fig = plt.figure(figsize=(16,4))

ax = fig.add_subplot(1,2,1)
ax.plot(history[0], lw=4)
ax.plot(history[1], lw=4)
ax.legend(['Train loss', 'Validation'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

if ENABLE_PLOT:
    ax=fig.add_subplot(1,2,2)
    ax.plot(history[2], lw=4)
    ax.plot(history[3], lw=4)
    ax.legend(['Train acc', 'Validation acc'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    plt.show()
    plt.plot

          
        

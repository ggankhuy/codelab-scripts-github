# x - input matrix, usually a bitmap image.
# w - kernel/filter = smaller moving array over x. 
# p - padding, extra spacing around original bitmap image.
# s - stride. Incremental move amount by kernel/filter.
# o - final result, feature map.

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import code
import numpy as np
#import PyQt5
from functools import wraps 
from datetime import datetime

def print_fcn_name(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string, ": ", func.__name__, " entered...")
        result = func(*args, **kwargs)
        return result

    return wrapper

print(matplotlib.get_backend())
#matplotlib.use("tkagg")

CONFIG_ENABLE_PLOT=0

image_path = "./"

transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform, download=True)
mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False, transform=transform, download=False)

batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

model=nn.Sequential()
model.to('cuda')
model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
model.add_module('ReLU1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
model.add_module('ReLU2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

#x=torch.ones((4,1,28,28), device='cuda')
#print(model(x).shape)

model.add_module('flatten', nn.Flatten())
model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('ReLU3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))
model.add_module('fc2', nn.Linear(1024,10))
model.add_module('softmax', nn.Softmax(dim=1))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

@print_fcn_name
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        print("\nEpoch: ", epoch, ": ", end=" ")
        for x_batch, y_batch in train_dl:
            print(".", end="")
            pred=model(x_batch)
            loss=loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        model.eval()

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred=model(x_batch)
                loss=loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += \
                    loss.item()*y_batch.size(0)
                is_correct=(torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'\nEpoch {epoch+1} accuracy: '
            f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
            f'{accuracy_hist_valid[epoch]:.4f}')

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

#code.interact(local=locals())

torch.manual_seed(1)
num_epochs=5
hist=train(model, num_epochs, train_dl, valid_dl)

if CONFIG_ENABLE_PLOT: 
    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(x_arr, hist[0], '-0', label='Train Loss')
    ax.plot(x_arr, hist[1], '--<', label='Validation Loss')

    ax.legend(fontsize=15)
    ax=fig.add_subplot(1,2,2)
    ax.plot(x_arr, hist[2], '-0', label='Train Acc')
    ax.plot(x_arr, hist[3], '--<', label='Validatin Acc')

    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()
    plt.plot
    plt.savefig("p474.plot1.jpg", dpi=300)

pred=model(mnist_test_dataset.data.unsqueeze(1) / 255. )
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')

'''
if CONFIG_ENABLE_PLOT:
    fig=plt.figure(figure=(12, 4))
   
    for i in range(12):
        ax=fig.add_subplot(2,6,i+1)
        ax.set_xticks([]); ax.set_yticks([])

        img=mnist_test_dataset[i][0][0,:,:]
        pred=model(img.unsqueeze(0).unsqueeze(1))
        y_pred = torch.argmax(pred)
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, y_pred.item(),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    plt.show()
'''

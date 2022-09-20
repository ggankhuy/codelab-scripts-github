import torch
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import code
import torch.nn as nn

learning_rate = 0.001
num_epochs = 200
log_epochs = 10

X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X_train, y_train)
plt.show()

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train)
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
print(train_dl)

loss_fn = nn.MSELoss(reduction='mean')
input_size = 1 
output_size = 1
model = nn.Linear(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

code.interact(local=locals())
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:

        #1. generate predictions.

        pred = model(x_batch)[:, 0]

        #2. calculate loss.

        loss = loss_fn(pred, y_batch)

        #3. compute gradients.

        loss.backward()

        #4. update parameters using gradients.

        optimizer.step()

        #5. reset the gradiants to zero.

        optimizer.zero_grad()

    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')    




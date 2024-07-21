import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

from torch.utils.data import DataLoader, Dataset,  random_split, TensorDataset
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('..')

from common.settings import *
from common.classes import *
from data_generation.square_sequences import generate_sequences

from stepbystep.v4 import StepByStep

from plots.chapter8 import plot_data

CONFIG_USE_SBS=0

print("Import setings:")
printDbg("hidden_dim: ", hidden_dim)
printDbg("n_features: ", n_features)

CONFIG_PLOT=0
points, directions = generate_sequences(256,  seed=13)
test_points, test_directions = generate_sequences(seed=19)

printTensor(points, globals())
printTensor(test_points, globals())

train_data = TensorDataset(torch.as_tensor(points).float(),torch.as_tensor(directions).view(-1,1).float())
test_data = TensorDataset(torch.as_tensor(test_points).float(), torch.as_tensor(test_directions).view(-1,1).float())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data,  batch_size=16)

'''
printTensor(train_data.data.numpy(), globals())
printTensor(test_data.data.numpy(), globals())
quit(0)
printTensor(train_loader, globals())
printTensor(test_loader, globals())
'''

torch.manual_seed(21)
model=SquareModel(n_features=n_features, hidden_dim=hidden_dim, n_outputs=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
writer = None
scheduler = None

if CONFIG_USE_SBS:
    sbs_rnn=StepByStep(model, loss, optimizer)
    sbs_rnn.set_loaders(train_loader, test_loader)
    sbs_rnn.train(100)
else:
    # We start by storing the arguments as attributes
    # to use them later

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    is_batch_lr_scheduler = False

    # These attributes are going to be computed internally
    losses = []
    val_losses = []
    learning_rates = []
    total_epochs = 0

    visualization = {}
    handles = {}

    def lr_range_test(data_loader, end_lr, num_iter=100, step_mode='exp', alpha=0.05, ax=None):
        # Since the test updates both model and optimizer we need to store
        # their initial states to restore them in the end
        previous_states = {'model': deepcopy(model.state_dict()),
                           'optimizer': deepcopy(optimizer.state_dict())}
        # Retrieves the learning rate set in the optimizer
        start_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # Builds a custom function and corresponding scheduler
        lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)

        # Variables for tracking results and iterations
        tracking = {'loss': [], 'lr': []}
        iteration = 0

    def make_lr_fn(start_lr, end_lr, num_iter, step_mode='exp'):
        if step_mode == 'linear':
            factor = (end_lr / start_lr - 1) / num_iter
            def lr_fn(iteration):
                return 1 + iteration * factor
        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / num_iter
            def lr_fn(iteration):
                return np.exp(factor)**iteration
        return lr_fn

    def epoch_schedulers(val_loss):
        if not is_batch_lr_scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            current_lr = list(map(lambda d: d['lr'], scheduler.optimizer.state_dict()['param_groups']))
            learning_rates.append(current_lr)

    def _mini_batch_schedulers(frac_epoch):
        if scheduler:
            if is_batch_lr_scheduler:
                if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(total_epochs + frac_epoch)
                else:
                    scheduler.step()

                current_lr = list(map(lambda d: d['lr'], scheduler.optimizer.state_dict()['param_groups']))
                learning_rates.append(current_lr)

    def _mini_batch(loss_fn, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        if validation:
            data_loader = val_loader
            step_fn = val_step_fn
        else:
            data_loader = train_loader
            step_fn = train_step_fn

        if data_loader is None:
            return None

        n_batches = len(data_loader)
        # Once the data loader and step function, this is the same
        # mini-batch loop we had before
        mini_batch_losses = []

        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            mini_batch_loss = step_fn(x_batch, y_batch, loss_fn)
            mini_batch_losses.append(mini_batch_loss)

            if not validation:
                _mini_batch_schedulers(i / n_batches)

        loss = np.mean(mini_batch_losses)
        return loss

    # Creates the train_step function for our model,
    # loss function and optimizer
    # Note: there are NO ARGS there! It makes use of the class
    # attributes directly

    def perform_val_step_fn(model, x, y, loss_fn):
        # Sets model to EVAL mode
        model.eval()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluati>
        return loss.item()

    def val_step_fn(model, x, y, loss, optimizer):
        # Sets model to EVAL mode
        model.eval()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # There is no need to compute Steps 3 and 4, since we don't update parameters during evaluation
        return loss.item()

    def train_step_fn(model, x, y, loss_fn, optimizer):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Computes our model's predicted output - forward pass
        yhat = model(x)
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y)
        # Step 3 - Computes gradients
        loss.backward()

        '''?!
        if callable(clipping):
            clipping()
        '''

        # Step 4 - Updates parameters using gradients and the learning rate
        optimizer.step()
        optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    # sbs.set_seed() substitute.

    lr_fn = make_lr_fn(start_lr, end_lr, num_iter)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_fn)

    seed=42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        train_loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass

    n_epochs = 100
    total_epochs = 0
    for epoch in range(n_epochs):
        # Keeps track of the numbers of epochs
        # by updating the corresponding attribute
        total_epochs += 1

        # inner loop
        # Performs training using mini-batches

        data_loader = train_loader
        step_fn = train_step_fn

        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and
        # corresponding step function is going to be used
        data_loader = train_loader
        step_fn = train_step_fn

        if data_loader:
            n_batches = len(data_loader)
            # Once the data loader and step function, this is the same
            # mini-batch loop we had before
            mini_batch_losses = []

            for i, (x_batch, y_batch) in enumerate(data_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                mini_batch_loss = step_fn(model, x_batch, y_batch, loss_fn, optimizer)
                mini_batch_losses.append(mini_batch_loss)

                validation = None
                if not validation:
                    _mini_batch_schedulers(i / n_batches)

            loss = np.mean(mini_batch_losses)

        losses.append(loss)

        # VALIDATION
        # no gradients in validation!
        with torch.no_grad():
            # Performs evaluation using mini-batches
            val_loader = test_loader
            data_loader = val_loader
            step_fn = val_step_fn

#           val_loss = _mini_batch(validation=True, loss_fn)
            # start def _mini_batch

            # The mini-batch can be used with both loaders
            # The argument `validation`defines which loader and
            # corresponding step function is going to be used

            data_loader = val_loader
            step_fn = val_step_fn

            if data_loader:
                n_batches = len(data_loader)
                # Once the data loader and step function, this is the same
                # mini-batch loop we had before
                mini_batch_losses = []

                for i, (x_batch, y_batch) in enumerate(data_loader):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)

                    mini_batch_loss = step_fn(model, x_batch, y_batch, loss_fn, optimizer)
                    mini_batch_losses.append(mini_batch_loss)

                    if not validation:
                        _mini_batch_schedulers(i / n_batches)

                loss = np.mean(mini_batch_losses)
                val_loss = loss 
    
            # end def _mini_batch()

            val_losses.append(val_loss)

        epoch_schedulers(val_loss)

        # If a SummaryWriter has been set...
        if writer:
            scalars = {'training': loss}
            if val_loss is not None:
                scalars.update({'validation': val_loss})
            # Records both losses for each epoch under the main tag "loss"
            writer.add_scalars(main_tag='loss',
                                    tag_scalar_dict=scalars,
                                    global_step=epoch)

        if writer:
            # Closes the writer
            writer.close()

if CONFIG_PLOT:
    fig=sbs_rnn.plot_losses()
    StepByStep.loader_apply(test_loader, sbs_rnn.correct)

state=model.basic_rnn.state_dict()
state['weight_ih_l0'], state['bias_ih_l0']
    


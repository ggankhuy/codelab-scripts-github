import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np
import sys
import time
import re
import helper

from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
DEBUG=0

CONFIG_ENABLE_PLOT=1
CONFIG_SAVE_MODEL=0

DEBUG=0
CONFIG_ENABLE_PLOT=0
CONFIG_EPOCHS=30
CONFIG_BATCH_SIZE=32

for i in sys.argv:
    print("Processing ", i)
    try:
        if re.search("epochs=", i):
            CONFIG_EPOCHS=int(i.split('=')[1])

        if re.search("batch_size=", i):
            CONFIG_BATCH_SIZE=int(i.split('=')[1])

    except Exception as msg:
        print("No argument provided, default values will be used.")

print("epochs: ", CONFIG_EPOCHS)
print("batch_size: ", CONFIG_BATCH_SIZE)

'''

if  len(sys.argv) > 1:
    CONFIG_EPOCHS, CONFIG_BATCH_SIZE = helper.process_params(sys.argv, ["epochs", "batch_size"])
'''

n_steps=50
series=helper.generate_series(10000, n_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

print("X_train, y_train shape: ", X_train.shape, y_train.shape)
print("X_valid, y_valid shape: ", X_valid.shape, y_valid.shape)
print("X_test, y_test shape:   ", X_test.shape, y_test.shape)

if CONFIG_ENABLE_PLOT:
    fig, ax = plt.subplots()
    ax.plot(series)

    plt.show()

y_pred = X_valid[:, -1]
print("y_pred shape: ", y_pred.shape)
print("test with basic pred: ", np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))


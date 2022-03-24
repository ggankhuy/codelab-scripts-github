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

CONFIG_ENABLE_PLOT=0
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
    '''
    fig, ax = plt.subplots()
    ax.plot(series)

    plt.show()
    '''

y_pred = X_valid[:, -1]
print("y_pred shape: ", y_pred.shape)
model=keras.models.Sequential([\
    keras.layers.SimpleRNN(1, input_shape=[None, 1])\
])

print("model summary: ", model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history=model.fit(X_train, y_train, epochs=CONFIG_EPOCHS, batch_size=CONFIG_BATCH_SIZE, validation_data=(X_valid, y_valid))

if CONFIG_ENABLE_PLOT:
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()

model.evaluate(X_test, y_test)

if DEBUG:
    print("model layers: ", model.layers)

#weights, biases  = model.layers[0].get_weights()

if DEBUG:
    print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape)

if CONFIG_SAVE_MODEL:
    model.save("p505.h5")
X_new = X_test[:3]
print("X_new shape: ", X_new.shape)
y_proba = model.predict(X_new)
print("y_proba (predict)(value): ", y_proba.round(2), "\ny_proba(shape)", np.array(y_proba).shape)

y_pred = model.predict_classes(X_new)
print("y_pred (predict_classes): ", y_pred)

mse_test = model.evaluate(X_test, y_test)
print("mse_test: ", mse_test)

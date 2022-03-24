# Using neural net to do a classification task.

import tensorflow as tf
import pandas as pd 
import matplotlib as plt
import sys 
import time
import re
import os
import numpy as np
import helper
from tensorflow import keras

DEBUG=0
CONFIG_ENABLE_PLOT=0
CONFIG_EPOCHS=30
CONFIG_BATCH_SIZE=32
CONFIG_SAVE_MODEL=1

CONFIG_SAVE_MODEL_MODE_KERAS=0
CONFIG_SAVE_MODEL_MODE_KERAS_H5=1
# this fill save with directory structure compatible with tf serving: p672.sh
CONFIG_SAVE_MODEL_MODE_SAVED_MODEL=2
CONFIG_SAVE_MODEL_MODE_CHECKPOINT=3
CONFIG_SAVE_MODEL_MODE=CONFIG_SAVE_MODEL_MODE_SAVED_MODEL

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

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("X_train_full/y_train_full/X_test/y_test: ", X_train_full.shape, y_train_full.shape, X_test.shape, y_test.shape)

SEPARATOR=10000
X_valid, X_train = X_train_full[:SEPARATOR] / 255.0, X_train_full[SEPARATOR:]/255.0
y_valid, y_train = y_train_full[:SEPARATOR], y_train_full[SEPARATOR:]
X_test = X_test / 255.0

print("X_valid/X_train/y_valid/y_train: ", X_valid.shape, X_train.shape, y_valid.shape, y_train.shape)

class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat" , "Sandal", "Shirt", "Sneaker","Bad","Ankle boot"]

model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(30, activation="softmax"))

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

weights, biases  = model.layers[1].get_weights()

if DEBUG:
    print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape) 

if CONFIG_SAVE_MODEL:
    print("CONFIG_SAVE_MODEL_MODE: ", CONFIG_SAVE_MODEL_MODE)
    if CONFIG_SAVE_MODEL_MODE == CONFIG_SAVE_MODEL_MODE_KERAS:
        print("CONFIG_SAVE_MODEL_MODE_KERAS...")
        model.save("p297.h5")
    elif CONFIG_SAVE_MODEL_MODE == CONFIG_SAVE_MODEL_MODE_SAVED_MODEL:
        print("CONFIG_SAVE_MODEL_SAVED_MODEL...")
        model_version="0001"
        model_name="p297"
        model_path=os.path.join(model_name, model_version)
        tf.saved_model.save(model, model_path)
    elif CONFIG_SAVE_MODEL_MODE == CONFIG_SAVE_MODEL_MODE_KERAS_H5:
        model.save("p297")
    elif CONFIG_SAVE_MODEL_MODE == CONFIG_SAVE_MODEL_CHECKPOINT:
        print("Checkpoint: Unimplemented...")
    else:
        print("Unknown option: ", CONFIG_SAVE_MODEL_MODE)
else:
    print("Saving model is not enabled. Not saving...")
X_new = X_test[:3]
print("X_new type: ", type(X_new))
np.save("test.npy", X_new)
print("X_new shape: ", X_new.shape)
y_proba = model.predict(X_new)
print("y_proba (predict)(value): ", y_proba.round(2), "\ny_proba(shape)", np.array(y_proba).shape)

# Deprecated with TF1.X: y_pred = model.predict_classes(X_new)

y_pred = np.argmax(y_proba,axis=1)

print("y_pred (predict_classes): ", y_pred)
print("y_test: ", y_test[:3])


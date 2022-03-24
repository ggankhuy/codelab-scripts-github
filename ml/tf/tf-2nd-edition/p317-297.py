# Using neural net to do a classification task.
# The difference is then log using tensorboard API.

import tensorflow as tf
import pandas as pd
import matplotlib as plt
import sys
import time
import re
import numpy as np
import helper
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
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

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# tensor board code.
#run_logdir = get_run_logdir(root_logdir, run_id)
tensorboard_cb = keras.callbacks.TensorBoard()

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("X_train_full.shape: ", X_train_full.shape)
print("X_train_full.dtype: ", X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test - X_test / 255.0
class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat" , "Sandal", "Shirt", "Sneaker","Bad","Ankle boot"]

model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(30, activation="softmax"))

print("model summary: ", model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history=model.fit(X_train, y_train, epochs=CONFIG_EPOCHS, batch_size=CONFIG_BATCH_SIZE, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])

pd.DataFrame(history.history).plot(figsize=(8, 5))

if CONFIG_ENABLE_PLOT:
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()

model.evaluate(X_test, y_test)

print("model layers: ", model.layers)
weights, biases  = model.layers[1].get_weights()
print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape) 
model.save("p297.h5")
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = model.predict_classes(X_new)
print("y_pred: ", y_pred)



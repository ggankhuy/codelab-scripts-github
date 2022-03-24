# Using neural net to do a classification task.
# This has similar objective as p297.py, the difference
# is p297.py saves the training model and this one load
# the model and predicts the one from p297.py

import tensorflow as tf
import pandas as pd 
import matplotlib as plt

from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

import onnx
from onnx_tf.backend import prepare

CONFIG_ENABLE_PLOT=0
CONFIG_LOAD_MODEL_ENABLE=1
CONFIG_LOAD_MODEL_TF=0
CONFIG_LOAD_MODEL_PT=1
CONFIG_LOAD_MODEL_ONNX=2
CONFIG_LOAD_MODEL=CONFIG_LOAD_MODEL_ONNX
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("X_train_full.shape: ", X_train_full.shape)
print("X_train_full.dtype: ", X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test - X_test / 255.0
class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat" , "Sandal", "Shirt", "Sneaker","Bad","Ankle boot"]

'''
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(30, activation="softmax"))

print("model summary: ", model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#history=model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
'''

if CONFIG_LOAD_MODEL_ENABLE:
    if CONFIG_LOAD_MODEL==CONFIG_LOAD_MODEL_ONNX:
        print("Loading onnx model...")
        model_import = onnx.load('output/p297.onnx')
        model = prepare(model_import)
    elif CONFIG_LOAD_MODEL==CONFIG_LOAD_MODEL_TF:
        print("Loading tf model...")
        model = keras.models.load_model("p297.h5")
    elif CONFIG_LOAD_MODEL==CONFIG_LOAD+MODEL_PT:
        print("Loading pt model...(not implemented)")
    else:
        print("Unknown CONFIG_LOAD_MODEL value: ", CONFIG_LOAD_MODEL)
    
'''
pd.DataFrame(history.history).plot(figsize=(8, 5))

if CONFIG_ENABLE_PLOT:
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()
'''

model.evaluate(X_test, y_test)

print("model layers: ", model.layers)
weights, biases  = model.layers[1].get_weights()
print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape) 
model.save("p297-2.h5")
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = model.predict_classes(X_new)
print("y_pred: ", y_pred)



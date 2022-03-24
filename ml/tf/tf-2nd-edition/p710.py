# Using neural net to do a classification task.

import tensorflow as tf
import pandas as pd 
import matplotlib as plt

from tensorflow import keras
print(tf.__version__)
print(keras.__version__)

CONFIG_ENABLE_PLOT=0

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print("X_train_full.shape: ", X_train_full.shape)
print("X_train_full.dtype: ", X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0
class_names = ["T-shirt/top","Trouser", "Pullover", "Dress", "Coat" , "Sandal", "Shirt", "Sneaker","Bad","Ankle boot"]

distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    mirrored_model=keras.models.Sequential()
    mirrored_model.add(keras.layers.Flatten(input_shape = [28, 28]))
    mirrored_model.add(keras.layers.Dense(300, activation="relu"))
    mirrored_model.add(keras.layers.Dense(100, activation="relu"))
    mirrored_model.add(keras.layers.Dense(30, activation="softmax"))

print("model summary: ", mirrored_model.summary())

mirrored_model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
batch_size=100
history=mirrored_model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))

if CONFIG_ENABLE_PLOT:
    plt.pyplot.grid(True)
    plt.pyplot.gca().set_ylim(0, 1)
    plt.pyplot.show()

mirrored_model.evaluate(X_test, y_test)

print("mirrored_model layers: ", mirrored_model.layers)
weights, biases  = mirrored_model.layers[1].get_weights()
print("weights, biases (shapes): ", weights, biases, weights.shape, biases.shape) 
mirrored_model.save("p710.h5")
X_new = X_test[:3]
y_proba = mirrored_model.predict(X_new)
print(y_proba.round(2))
y_pred = np.argmax(y_proba,axis=1)
print("y_pred (predict_classes): ", y_pred)
print("y_test: ", y_test[:3])




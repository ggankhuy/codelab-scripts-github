# using neural net for regression task. 
# This is similar to p308.py except p308.py uses fully connected sequential neural net whereas
# this one uses nonsequential network. But the data itself does not use or take adv. of 
# non-sequential network.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

print("training data shapes:")
print("X_train_full/test/t_train_full/test: ", X_train_full.shape, X_test.shape, y_train_full.shape, y_test.shape)
print("X_train/X_valid/y_train/y_valid: ", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
print("training result (shape): ", history)
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3] # prertend these are new instances.
y_preid = model.predict(X_new)


# using neural net for regression task.
# It uses the p320 features, including fine-tuning the neural network
# hyperparameters to tweak.
# This is similar to p320-308.py but instead of using single,default 
# parameters specified in build_mode(), it defines param_distribs to 
# to define permutations of parameters.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import time 

TEST_MODE=1

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

param_distribs = {\
    "n_hidden" : [0,1,2,3],
    "n_neurons" : np.arange(90, 100, 2),
    "learning_rate" : reciprocal(3e-4, 3e-2),
}

print("param_distribs: ", param_distribs)
time.sleep(5)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    print("----------------------")
    print("n_hidden: ", n_hidden, ", n_neurons: ", n_neurons, ", learning_rate: ", learning_rate, ", input_shape: ", input_shape)
    print("----------------------")
    model=keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer1 in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

reg_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
if TEST_MODE:
    reg_search_cv.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), \
        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
else:
    reg_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), \
        callbacks=[keras.callbacks.EarlyStopping(patience=10)])

'''
if TEST_MODE:
    keras_reg.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), \
    callbacks=[keras.callbacks.EarlyStopping(patience=10)])
else:
    keras_reg.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), \
    callbacks=[keras.callbacks.EarlyStopping(patience=10)])
'''
X_new = X_test[:3] # pretend these are new instances.
mse_test = keras_reg.score(X_test, y_test)
print("mse_test: ", mse_test)
y_pred = keras_reg.predict(X_new)
print("y_pred: ", y_pred)


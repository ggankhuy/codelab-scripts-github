# using neural net for regression task.
# It uses the p320 features, including fine-tuning the neural network
# hyperparameters to tweak.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
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

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model=keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer1 in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

if TEST_MODE:
    keras_reg.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), \
    callbacks=[keras.callbacks.EarlyStopping(patience=10)])
else:
    keras_reg.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), \
    callbacks=[keras.callbacks.EarlyStopping(patience=10)])

X_new = X_test[:3] # prertend these are new instances.
mse_test = keras_reg.score(X_test, y_test)
print("mse_test: ", mse_test)
y_pred = keras_reg.predict(X_new)
print("y_pred: ", y_pred)


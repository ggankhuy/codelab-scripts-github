# using neural net for regression task. 
# This is similar to p310.py, a non-sequential network however the difference
# is implementation of separate output. 

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

print("X_train/y_train/full/test shapes: ", X_train_full.shape, X_test.shape, y_train_full.shape, y_test.shape)
print("X_/y_/train/valid: ", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test[:3, 2:]

print("X_train_A/B/valid_A/B: ", X_train_A.shape, X_train_B.shape, X_valid_A.shape, X_valid_B.shape)
print("X_test_A/B/new_A/B: ", X_test_A.shape, X_test_B.shape, X_new_A.shape, X_new_B.shape)

input_A = keras.layers.Input(shape=[5] , name="wide_input")
input_B = keras.layers.Input(shape=[6] , name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
main_output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], outputs=[main_output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
#model.compile(loss=["mse","mse"], loss_weights=[0.9, 0.1], optimizer="sgd")

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20, validation_data=([X_valid_A, X_valid_B],[y_valid, y_valid]))
print("training result (shape): ", history)
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])

print("Losses: total, main, aux: ", total_loss, main_loss, aux_loss)

y_pred_main, y_pred_aux = model.predict(( \
    [X_new_A,X_new_B]))

print("prediction: ", y_pred_main, y_pred_aux)



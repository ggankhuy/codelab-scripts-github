# stateless RNN processing NLP.

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
CONFIG_EPOCHS=20
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

shakespeare_url="https://homl.info/shakespeare"
filepath=keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text=f.read()
tokenizer = keras.preprocessing.text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(shakespeare_text)

max_id = len(tokenizer.word_index)
dataset_size = tokenizer.document_count
[encoded]=np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])

n_steps=100
window_length = n_steps + 1
dataset = dataset.window(window_length, shift=1, drop_remainder = True)
dataset = dataset.flat_map(lambda windows: windows.batch(window_length))

batch_size=CONFIG_BATCH_SIZE
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)

distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    model=keras.models.Sequential([\
        keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], \
            dropout=0.2, recurrent_dropout=0.2),\
        keras.layers.GRU(128, return_sequences=True, \
            dropout=0.2, recurrent_dropout=0.2),\
        keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))\
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history=model.fit(dataset, epochs=CONFIG_EPOCHS)

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are you"])
Y_pred = model.predict_classes(X_new)
print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])


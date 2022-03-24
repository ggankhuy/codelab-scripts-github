# Sentimental analysis.

import tensorflow as tf
import pandas as pd
import matplotlib as plt
import numpy as np
import sys
import time
import re
import helper
import tensorflow_datasets as tfds

from collections import Counter
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)
DEBUG=0

CONFIG_ENABLE_PLOT=0
CONFIG_SAVE_MODEL=0

DEBUG=0
CONFIG_ENABLE_PLOT=0
CONFIG_EPOCHS=5
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

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
print("X_train[:10]: ", X_train[0][:10])

word_index = keras.datasets.imdb.get_word_index()
id_to_word = {id_ +  3: word for word, id_ in word_index.items()}
for id_,  token in enumerate(("<pad>", "<sos>", "<unk>")):
    id_to_word[id_] = token

" ".join([id_to_word[id_] for id in X_train[0][:10]])

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info = True)
train_size = info.splits["train"].num_examples

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, b">br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

print("most common words in vocabulary: ", vocabulary.most_common()[:3])

vocab_size = 10000
truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]

words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init=tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

embed_size = 128
'''
with distribution.scope():

    mirrored_model=keras.models.Sequential()
    mirrored_model.add(keras.layers.Flatten(input_shape = [28, 28]))
    mirrored_model.add(keras.layers.Dense(300, activation="relu"))
    mirrored_model.add(keras.layers.Dense(100, activation="relu"))
    mirrored_model.add(keras.layers.Dense(30, activation="softmax"
'''
distribution = tf.distribute.MirroredStrategy()

with distribution.scope():
    model=keras.models.Sequential([\
        keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size, input_shape=[None]), \
        keras.layers.GRU(128, return_sequences=True),\
        keras.layers.GRU(128),\
        keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history=model.fit(train_set, epochs=CONFIG_EPOCHS)

'''
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are you"])
Y_pred = model.predict_classes(X_new)
print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1])
'''

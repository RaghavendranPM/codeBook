# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.engine.topology import Merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import keras.backend as K

vocab_size = 5000
embed_size = 300

word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,
                         init="glorot_uniform",
                         input_length=1))
word_model.add(Reshape((embed_size,)))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,
                            init="glorot_uniform",
                            input_length=1))
context_model.add(Reshape((embed_size,)))

model = Sequential()
model.add(Merge([word_model, context_model], mode="dot"))
model.add(Dense(1, init="glorot_uniform", activation="sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam")

merge_layer = model.layers[0]
word_model = merge_layer.layers[0]
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0]

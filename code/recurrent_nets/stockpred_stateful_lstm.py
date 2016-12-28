# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import re
import os

DATA_DIR = "../data"

NUM_TIMESTEPS = 4
HIDDEN_SIZE = 4

BATCH_SIZE = 5
NUM_EPOCHS = 5

fprices = open(os.path.join(DATA_DIR, "AirQualityUCI.csv"), "rb")
data = []
for line in fprices:
    if line.startswith("#") or line.startswith(";;;;"):
        continue
    cols = line.strip().split(";")
    cogt = float(re.sub(",", ".", cols[2]))
    data.append(float(cogt))

fprices.close()

data = np.array(data, dtype="float32").reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
scaler.fit_transform(data)

# transform to 4 inputs -> 1 label format
X = np.zeros((len(data), 4))
Y = np.zeros((len(data), 1))
for i in range(len(data) - 5):
    X[i] = data[i:i + 4].T
    Y[i] = data[i + 5]

# reshape to three dimensions (samples, timesteps, features)
X = np.expand_dims(X, axis=2)

# split into training and test sets (add the extra offsets so 
# we can use batch size of 5)
sp = int(0.7 * len(data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

## stateless
#model = Sequential()
#model.add(LSTM(4, input_shape=(NUM_TIMESTEPS, 1), 
#               return_sequences=False))
#model.add(Dense(1))
## /stateless

# stateful
model = Sequential()
model.add(LSTM(4, stateful=True,
               batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 1), 
               return_sequences=False))
model.add(Dense(1))
# /stateful

model.compile(loss="mean_squared_error", optimizer="adam",
              metrics=["mean_squared_error"])

## stateless
#model.fit(Xtrain, Ytrain, nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE,
#          validation_data=(Xtest, Ytest))
## /stateless
          
# stateful
# need to make training and test data to multiple of BATCH_SIZE
train_size = (Xtrain.shape[0] // BATCH_SIZE) * BATCH_SIZE
test_size = (Xtest.shape[0] // BATCH_SIZE) * BATCH_SIZE
Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

for i in range(NUM_EPOCHS):
    print("Epoch {:d}/{:d}".format(i, NUM_EPOCHS))
    model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(Xtest, Ytest),
              shuffle=False)
    model.reset_states()
# /stateful

rmse = math.sqrt(model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)[0])
print("\nRMSE: {:.3f}".format(rmse))

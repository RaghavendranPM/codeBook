import numpy as np
np.random.seed(1111)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.utils.visualize_util import plot

VERBOSE=1
# Output classes, number of MINST DIGITS
NUM_CLASSES = 10
# Shape of an MINST digit image
SHAPE_X, SHAPE_Y = 28, 28
# Channels on MINST
IMG_CHANNELS = 1

# Network Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 12
# Number of convolutional filters 
NUM_FILTERS = 32
# side length of maxpooling square
NUM_POOL = 2
# side length of convolution square
NUM_CONV = 3
# dropout rate for regularization
DROPOUT_RATE = 0.5
# hidden number of neurons first layer
N_HIDDEN = 128
# validation data
VALIDATION_SPLIT=0.2 # 20%

# LOAD the MINST DATA split in training and test data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, SHAPE_X, SHAPE_Y)
X_test = X_test.reshape(X_test.shape[0], 1, SHAPE_X, SHAPE_Y)

# convert in float32 representation for GPU computation
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# NORMALIZE each pixerl by dividing by max_value=255
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
 
# KERAS needs to represent each output class into OHE representation
Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

nn = Sequential()
 
#FIRST LAYER OF CONVNETS, POOLING, DROPOUT
#  apply a NUM_CONV x NUM_CONF convolution with NUM_FILTERS output
#  for the first layer it is also required to define the input shape
#  activation function is rectified linear 
nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV, 
	input_shape=(IMG_CHANNELS, SHAPE_X, SHAPE_Y) ))
nn.add(Activation('relu'))
nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV))
nn.add(Activation('relu'))
nn.add(MaxPooling2D(pool_size = (NUM_POOL, NUM_POOL)))
nn.add(Dropout(DROPOUT_RATE))

#SECOND LAYER OF CONVNETS, POOLING, DROPOUT 
#  apply a NUM_CONV x NUM_CONF convolution with NUM_FILTERS output
nn.add(Convolution2D( NUM_FILTERS, NUM_CONV, NUM_CONV))
nn.add(Activation('relu'))
nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV))
nn.add(Activation('relu'))
nn.add(MaxPooling2D(pool_size = (NUM_POOL, NUM_POOL) ))
nn.add(Dropout(DROPOUT_RATE))
 
# FLATTEN the shape for dense connections 
nn.add(Flatten())
 
# FIRST HIDDEN LAYER OF DENSE NETWORK
nn.add(Dense(N_HIDDEN))  
nn.add(Activation('relu'))
nn.add(Dropout(DROPOUT_RATE))          

# OUTFUT LAYER with NUM_CLASSES OUTPUTS
# ACTIVATION IS SOFTMAX, REGULARIZATION IS L2
nn.add(Dense(NUM_CLASSES, W_regularizer=l2(0.01) ))
nn.add(Activation('softmax') )

#summary
nn.summary()
#plot the model
plot(nn)

# COMPILE THE MODEL
#   loss_function is categorical_crossentropy
#   optimizer is adam
nn.compile(loss='categorical_crossentropy', 
	optimizer='adam', metrics=["accuracy"])

# FIT THE MODEL WITH VALIDATION DATA
nn.fit(X_train, Y_train, \
	batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, \
	verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Test the network
results = nn.evaluate(X_test, Y_test, verbose=VERBOSE)
print('accuracy:', results[1])

'''Trains a simple binarize CNN on the FMNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.98% test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
#np.random.seed(1337)  # for reproducibility

import keras
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers import Flatten
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Model

#from binary_ops import binary_tanh as binary_tanh_op
#from binary_layers import BinaryDense, BinaryConv2D

from keras.utils import plot_model
#from keras.utils import model_to_dot

from IPython.display import SVG
import graphviz
import pydot_ng as pydot

##################################################################################################################################
# Def new functions
def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


##################################################################################################################################
H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 12
channels = 1
img_rows = 28
img_cols = 28
filters = 32
kernel_size = (3, 3)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

##################################################################################################################################
# the data, shuffled and split between train_m and test_m sets
(X_train_m, y_train_m), (X_test_m, y_test_m) = mnist.load_data()

X_train_m = X_train_m.reshape(60000, 1, 28, 28)
X_test_m = X_test_m.reshape(10000, 1, 28, 28)
X_train_m = X_train_m.astype('float32')
X_test_m = X_test_m.astype('float32')
X_train_m /= 255
X_test_m /= 255
print(X_train_m.shape[0], 'train_m samples')
print(X_test_m.shape[0], 'test_m samples')

# convert class vectors to binary class matrices
#Y_train_m = np_utils.to_categorical(y_train_m, classes) * 2 - 1 # -1 or 1 for hinge loss
#Y_test_m = np_utils.to_categorical(y_test_m, classes) * 2 - 1

Y_train_m = np_utils.to_categorical(y_train_m, classes)
Y_test_m  = np_utils.to_categorical(y_test_m, classes)

##################################################################################################################################
##################################################################################################################################
# model generation
merged1 = load_model('mnist_baseline.h5', custom_objects={'relu6': relu6})
##################################################################################################################################
# model compilation
#opt = Adam()
#merged1.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

##################################################################################################################################
# Test Scoring
score1 = merged1.evaluate(X_test_m, Y_test_m, verbose=0)

##################################################################################################################################
# model details
merged1.summary()

print('Test score baseline mnist:', score1[0])
print('Test accuracy baseline mnist:', score1[1])

##################################################################################################################################

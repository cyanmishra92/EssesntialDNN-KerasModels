'''Trains a simple binarize CNN on the FMNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 98.98% test accuracy after 20 epochs using tensorflow backend
'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import keras
import keras.backend as K
import graphviz
import pydot_ng as pydot
#np.random.seed(1337)  # for reproducibility

# model prep
from keras.datasets            import cifar10
from keras.preprocessing.image import ImageDataGenerator

from keras.models              import Model
from keras.models              import Sequential
from keras.models              import load_model
from keras.layers              import Input, Dense, Dropout, Activation
from keras.layers              import BatchNormalization, MaxPooling2D
from keras.layers              import Flatten, AveragePooling2D, Conv2D

# callbacks & optimizers
from keras                     import optimizers
from keras.optimizers          import SGD, Adam, RMSprop
from keras.callbacks           import LearningRateScheduler, EarlyStopping
from keras.callbacks           import CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils               import np_utils
from keras.utils               import plot_model
from keras.utils               import plot_model
from keras.metrics             import top_k_categorical_accuracy

from IPython.display           import SVG

#from binary_ops import binary_tanh as binary_tanh_op
#from binary_layers import BinaryDense, BinaryConv2D

#from keras.utils import model_to_dot

##################################################################################################################################
# Def new functions
def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

##################################################################################################################################
H                        = 1.
kernel_lr_multiplier     = 'Glorot'

# nn
batch_size               = 32
epochs                   = 20
total_epochs             = 21
channels                 = 3
img_rows                 = 32
img_cols                 = 32
filters                  = 32
kernel_size              = (3, 3)
pool_size                = (2, 2)
hidden_units             = 128
classes                  = 10
use_bias                 = False

# learning rate schedule
lr_start                 = 1e-3
lr_end                   = 1e-4
lr_decay                 = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon                  = 1e-6
momentum                 = 0.9

# dropout
p1                       = 0.25
p2                       = 0.5

##################################################################################################################################
# the data, shuffled and split between train_f and test_f sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train   = X_train.astype('float32')
X_test    = X_test.astype('float32')
X_train  /= 255
X_test   /= 255

Y_train = keras.utils.to_categorical(y_train, classes)
Y_test = keras.utils.to_categorical(y_test, classes)
# convert class vectors to binary class matrices
##################################################################################################################################
# model generation
cfar10M = load_model('cifar10_baseline.h5', custom_objects={'relu6': relu6, 'top_3_accuracy': top_3_accuracy})
cfar10S = load_model('read_cifar10_baseline.h5', custom_objects={'relu6': relu6, 'top_3_accuracy': top_3_accuracy})

##################################################################################################################################
# model details
cfar10M.summary()
cfar10S.summary()

##################################################################################################################################
# Test Scoring
score_cfar10b = cfar10M.evaluate(X_test, Y_test, verbose=0)
score_cfar10s = cfar10S.evaluate(X_test, Y_test, verbose=0)
#score3 = combMdl.evaluate([X_test_m, X_test], [Y_test_m, Y_test], verbose=0)

##################################################################################################################################
# model details

print('Test score cifar10 baseline:',    score_cfar10b[0])
print('Test accuracy cifar10 baseline:', score_cfar10b[1])

print('Test score cifar10 modified:',    score_cfar10s[0])
print('Test accuracy cifar10 modified:', score_cfar10s[1])

##################################################################################################################################

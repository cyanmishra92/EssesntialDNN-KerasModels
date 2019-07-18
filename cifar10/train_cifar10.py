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
K.set_floatx('float16')
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
lr_start                 = np.float16(1e-3)
lr_end                   = np.float16(1e-4)
lr_decay                 = np.float16((lr_end / lr_start)**(1. / epochs))

# BN
epsilon                  = np.float16(1e-6)
momentum                 = np.float16(0.9)

# dropout
p1                       = 0.25
p2                       = 0.5

##################################################################################################################################
# the data, shuffled and split between train_f and test_f sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train   = X_train.astype('float16')
X_test    = X_test.astype('float16')
X_train  /= 255
X_test   /= 255

Y_train = keras.utils.to_categorical(y_train, classes)
Y_test = keras.utils.to_categorical(y_test, classes)
# convert class vectors to binary class matrices
##################################################################################################################################
# Model Building layers
input   = Input(shape=(32,32,3))
input_d = Input(shape=(1,28,28))

# conv11
conv11  = Conv2D(32, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',
          dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None, input_shape=X_train.shape[1:], name='conv11')(input)
bn11    = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn11')(conv11)
act11   = Activation(relu6, name='act11')(bn11)

# conv12
conv12  = Conv2D(32, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',
          dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None, name='conv12')(act11)
pool12  = MaxPooling2D(pool_size=pool_size, name='pool12', data_format='channels_last')(conv12)
bn12    = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn12')(pool12)
act12   = Activation(relu6, name='act12')(bn12)

# dropout11
dout11  = Dropout(0.25)(act12)

# conv13
conv13  = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',
          dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None, name='conv13')(dout11)
bn13    = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn13')(conv13)
act13   = Activation(relu6, name='act13')(bn13)

# conv14
conv14  = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',
          dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
          kernel_constraint=None, bias_constraint=None, name='conv14')(act13)
pool14  = MaxPooling2D(pool_size=pool_size, name='pool14', data_format='channels_last')(conv14)
bn14    = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn14')(pool14)
act14   = Activation(relu6, name='act14')(bn14)

# dropout12
dout11  = Dropout(0.25)(act14)


#flatten11
flat11  = Flatten()(dout11)

#Dense_11
dns11   = Dense(512, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
          bias_constraint=None, name='dns11')(flat11)
d_bn11  = BatchNormalization(epsilon=epsilon, momentum=momentum, name='d_bn11')(dns11)
d_act11 = Activation(relu6, name='d_act11')(d_bn11)

# dropout13
dout13  = Dropout(0.5)(d_act11)
#Dense_12
dns12   = Dense(classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
          kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
          bias_constraint=None, name='dns12')(d_act11)
d_bn12  = BatchNormalization(epsilon=epsilon, momentum=momentum, name='d_bn12')(dns12)
d_act12 = Activation('softmax', name='d_act12')(d_bn12)

##################################################################################################################################
# model generation
cfar10M = Model(inputs=[input],outputs=[d_act12])

##################################################################################################################################
# model compilation
opt     = Adam()
earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
cfar10M.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])#, top_3_accuracy])

#cfar10M.compile(loss= keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adadelta(), metrics=['accuracy'])

##################################################################################################################################
# model details
cfar10M.summary()

##################################################################################################################################
# model visualization
plot_model(cfar10M, to_file='cfar10_baseline.png', show_shapes=True, show_layer_names=True)

##################################################################################################################################
# model fit

##### Begin Batch Train #####

#lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
cfarlog       = 'training_cifar10' + '_baseline_' + '.log'
csv_logger    = CSVLogger(cfarlog)
reduce_lr     = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                        patience=5, min_lr=0.0001)
tbCallBack    = keras.callbacks.TensorBoard(log_dir='./Graph_cfar10', histogram_freq=0, write_graph=True, write_images=True)

print ("############### Training CIFAR10 Dataset ################")
history1      = cfar10M.fit(X_train, Y_train,
                     batch_size=batch_size, epochs=epochs,
                     verbose=1, validation_data=(X_test, Y_test),
                     callbacks=[reduce_lr, csv_logger, tbCallBack])

##### End Batch Train #####

##################################################################################################################################
# Test Scoring
score_cfar10b = cfar10M.evaluate(X_test, Y_test, verbose=0)
#score3 = combMdl.evaluate([X_test_m, X_test], [Y_test_m, Y_test], verbose=0)

##################################################################################################################################
# model details
cfar10M.summary()

print('Test score cifar10 baseline:',    score_cfar10b[0])
print('Test accuracy cifar10 baseline:', score_cfar10b[1])

##################################################################################################################################
## model saving
cfar10M.save('cifar10_baseline.h5')
cfar10M.save('to_drop_layers_cifar10_baseline.h5')
cfar10M.save('golden_cifar10_baseline.h5')
cfar10M.save('read_cifar10_baseline.h5')
#combMdl.save('combMdl.h5')

##################################################################################################################################

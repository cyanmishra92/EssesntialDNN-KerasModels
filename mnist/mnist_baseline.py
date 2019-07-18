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
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, AveragePooling2D, Conv2D
from keras.layers import Flatten
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy

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

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

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
input   = Input(shape=(1,28,28))

# conv1
conv1 = Conv2D(32, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv1')(input)
pool21= MaxPooling2D(pool_size=pool_size, name='pool21', data_format='channels_first')(conv1)
bn1   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1')(pool21)
act1  = Activation(relu6, name='act1')(bn1)

# conv2
conv2 = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv2')(act1)
pool22= MaxPooling2D(pool_size=pool_size, name='pool22', data_format='channels_first')(conv2)
bn2   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2')(pool22)
act2  = Activation(relu6, name='act2')(bn2)


# conv3
conv3 = Conv2D(64, kernel_size, strides=(1, 1), padding='valid', data_format='channels_first',
        dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, name='conv3')(act2)
pool23= MaxPooling2D(pool_size=pool_size, name='pool23', data_format='channels_first')(conv3)
bn3   = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3')(pool23)
act3  = Activation(relu6, name='act3')(bn3)

# avg pool
#apool1 = AveragePooling2D(pool_size=(5,5), name='apool1', data_format='channels_first')(pool23)

#flatten_1
flat1 = Flatten()(act3)

#Dense_11
dns11 = Dense(64, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, name='dns11')(flat1)
bn_d1 = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn_d1')(dns11)
act_d1= Activation(relu6, name='act_d1')(bn_d1)

#Dense_12
dns12 = Dense(classes, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None, name='dns12')(act_d1)
bn_d2 = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn_d2')(dns12)
dro   = Dropout(0.01)(bn_d2)
act_d2= Activation('softmax', name='act_d2')(dro)

##################################################################################################################################
# model generation
merged1 = Model(inputs=[input],outputs=[act_d2])

##################################################################################################################################
# model compilation
#opt = Adam(lr=lr_start)
opt = Adam()
earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
merged1.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy', top_3_accuracy])
#merged1.compile(loss= keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adadelta(), metrics=['accuracy'])

##################################################################################################################################
# model details
merged1.summary()

##################################################################################################################################
# model visualization
plot_model(merged1, to_file='merged1.png', show_shapes=True, show_layer_names=True)

##################################################################################################################################
# model fit

##### Begin Batch Train #####

#lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
mnist         = 'training_mnist'  + '_baseline_' + '.log'
csv_logger_m  = CSVLogger(mnist)
reduce_lr_m   = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=5, min_lr=0.0001)
tbCallBack_m  = keras.callbacks.TensorBoard(log_dir='./Graph_m', histogram_freq=0, write_graph=True, write_images=True)

print ("#################### Training Number MNIST Dataset ####################")
history1 = merged1.fit(X_train_m, Y_train_m,
                batch_size=batch_size, epochs=epochs,
                verbose=1, validation_data=(X_test_m, Y_test_m),
                callbacks=[reduce_lr_m, csv_logger_m, tbCallBack_m])

##### End Batch Train #####

##################################################################################################################################
# Test Scoring
score1 = merged1.evaluate(X_test_m, Y_test_m, verbose=0)

##################################################################################################################################
# model details
merged1.summary()

print('Test score baseline mnist:', score1[0])
print('Test accuracy baseline mnist:', score1[1])

##################################################################################################################################
## model saving
merged1.save('mnist_baseline.h5')

##################################################################################################################################

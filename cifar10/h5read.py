import h5py as h5
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Config
change_layer_1 = 0
change_layer_2 = 0
change_layer_3 = 0
change_layer_4 = 1


# Read H5 file
f = h5.File("read_cifar10_baseline.h5", "r+")
data = f['/model_weights/conv11/conv11/kernel:0']
print(data.shape)
print("changing weights and bias")
idty_base = np.identity(3)
idty      = np.repeat(idty_base, 3, axis=1).reshape((3,3,3))

#Layer 1
if (change_layer_1 == 0):

    f['/model_weights/conv11/conv11/kernel:0'][:,:,:,3 ]      = 0
    f['/model_weights/conv11/conv11/kernel:0'][:,:,:,23]      = 0
    f['/model_weights/conv11/conv11/kernel:0'][:,:,:,9 ]      = 0
    f['/model_weights/conv11/conv11/kernel:0'][:,:,:,10]      = 0
    #f['/model_weights/conv11/conv11/kernel:0'][:,:,:,5 ]      = 0
    #f['/model_weights/conv11/conv11/kernel:0'][:,:,:,17]      = 0
    #f['/model_weights/conv11/conv11/kernel:0'][:,:,:,0 ]      = 0
    #f['/model_weights/conv11/conv11/kernel:0'][:,:,:,11]      = 0



    f['/model_weights/conv11/conv11/bias:0'][3 ]              = 0
    f['/model_weights/conv11/conv11/bias:0'][23]              = 0
    f['/model_weights/conv11/conv11/bias:0'][9 ]              = 0
    f['/model_weights/conv11/conv11/bias:0'][10]              = 0
    #f['/model_weights/conv11/conv11/bias:0'][5 ]              = 0
    #f['/model_weights/conv11/conv11/bias:0'][17]              = 0
    #f['/model_weights/conv11/conv11/bias:0'][0 ]              = 0
    #f['/model_weights/conv11/conv11/bias:0'][11]              = 0

# Layer 2
if (change_layer_2 == 0):

    f['/model_weights/conv12/conv12/kernel:0'][:,:,:,14]      = 0
    f['/model_weights/conv12/conv12/kernel:0'][:,:,:,13]      = 0
    f['/model_weights/conv12/conv12/kernel:0'][:,:,:,25]      = 0
    f['/model_weights/conv12/conv12/kernel:0'][:,:,:,28]      = 0
    #f['/model_weights/conv12/conv12/kernel:0'][:,:,:,1 ]      = 0
    #f['/model_weights/conv12/conv12/kernel:0'][:,:,:,23]      = 0
    #f['/model_weights/conv12/conv12/kernel:0'][:,:,:,30]      = 0
    #f['/model_weights/conv12/conv12/kernel:0'][:,:,:,19]      = 0



    f['/model_weights/conv12/conv12/bias:0'][14]              = 0
    f['/model_weights/conv12/conv12/bias:0'][13]              = 0
    f['/model_weights/conv12/conv12/bias:0'][25]              = 0
    f['/model_weights/conv12/conv12/bias:0'][28]              = 0
    #f['/model_weights/conv12/conv12/bias:0'][1 ]              = 0
    #f['/model_weights/conv12/conv12/bias:0'][23]              = 0
    #f['/model_weights/conv12/conv12/bias:0'][30]              = 0
    #f['/model_weights/conv12/conv12/bias:0'][19]              = 0

# Layer 3
if (change_layer_3 == 0):

    f['/model_weights/conv13/conv13/kernel:0'][:,:,:,33]      = 0
    f['/model_weights/conv13/conv13/kernel:0'][:,:,:,11]      = 0
    f['/model_weights/conv13/conv13/kernel:0'][:,:,:,27]      = 0
    f['/model_weights/conv13/conv13/kernel:0'][:,:,:,48]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,25]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,26]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,0 ]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,4 ]      = 0

    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,22]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,29]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,34]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,12]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,45]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,7 ]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,56]      = 0
    #f['/model_weights/conv13/conv13/kernel:0'][:,:,:,44]      = 0


    f['/model_weights/conv13/conv13/bias:0'][33]              = 0
    f['/model_weights/conv13/conv13/bias:0'][11]              = 0
    f['/model_weights/conv13/conv13/bias:0'][27]              = 0
    f['/model_weights/conv13/conv13/bias:0'][48]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][25]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][26]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][0 ]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][4 ]              = 0

    #f['/model_weights/conv13/conv13/bias:0'][22]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][29]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][34]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][12]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][45]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][7 ]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][56]              = 0
    #f['/model_weights/conv13/conv13/bias:0'][44]              = 0


# Layer 4
if (change_layer_4 == 0):

    f['/model_weights/conv14/conv14/kernel:0'][:,:,:,53]      = 0
    f['/model_weights/conv14/conv14/kernel:0'][:,:,:,56]      = 0
    f['/model_weights/conv14/conv14/kernel:0'][:,:,:,34]      = 0
    f['/model_weights/conv14/conv14/kernel:0'][:,:,:,9 ]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,38]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,59]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,35]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,27]      = 0

    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,63]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,50]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,18]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,2 ]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,54]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,28]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,58]      = 0
    #f['/model_weights/conv14/conv14/kernel:0'][:,:,:,3 ]      = 0


    f['/model_weights/conv14/conv14/bias:0'][53]              = 0
    f['/model_weights/conv14/conv14/bias:0'][56]              = 0
    f['/model_weights/conv14/conv14/bias:0'][34]              = 0
    f['/model_weights/conv14/conv14/bias:0'][9 ]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][38]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][59]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][35]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][27]              = 0

    #f['/model_weights/conv14/conv14/bias:0'][63]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][50]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][18]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][2 ]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][54]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][28]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][58]              = 0
    #f['/model_weights/conv14/conv14/bias:0'][3 ]              = 0

f.close()

print("changed weight and bias")

print("loading updated data")
f = h5.File("read_cifar10_baseline.h5", "r+")
#datasetNames  = [n for n in f.keys()]
#for n in datasetNames:
#    print(n)
#print(f)
#model_weights = f['model_weights']
#conv11        = model_weights['conv11']
#kernel        = conv11['conv11']
#kernel_w      = kernel['kernel:0']
##kernel11      = np.array(kernel_w)
#print(kernel_w.shape)
##kernel_w[0,0,0,1]=1
#for i in range(0, 32):
#    print(i, " ---> ",  LA.norm(kernel_w[:,:,:,i]))
print("validated new file. Running tests...")
f.close()

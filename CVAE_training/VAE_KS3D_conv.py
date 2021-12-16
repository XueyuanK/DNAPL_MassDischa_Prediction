import numpy as np
import os
os.system('module load cuda/9.0')
os.system('module load cudnn/9.0_v7.6.4')
import copy
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from scipy.io import savemat, loadmat
from keras.losses import mse,sparse_categorical_crossentropy
from keras.layers import Add, Reshape, Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose,Conv3D, Conv3DTranspose,Dropout
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
K.set_image_data_format('channels_last') 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import Callback
from keras import initializers
import hdf5storage

# total number of epochs
n_epochs = 100 

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight 

    def on_epoch_end (self, epoch, logs={}):
        def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
            L = np.ones(n_epoch)
            period = n_epoch/n_cycle
            step = (stop-start)/(period*ratio) # linear schedule
            for c in range(n_cycle):
                v , i = start , 0
                while v <= stop and (int(i+c*period) < n_epoch):
                    L[int(i+c*period)] = v
                    v += step
                    i += 1
            return L
        #new_weight= frange_cycle_linear(0.0,1.0,n_epochs,4)
        new_weight=1.0
        #K.set_value(self.weight, new_weight[epoch])
        K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))

def sampling(input_param):
    """
    sampling the latent space from a Gaussian distribution:
    # Input
        input_param: mean and log of variance of q(z|x)
    # Output
        z: sampled latent space vector
    """
    #mean and log(var):
    z_mean, z_log_var = input_param
    #dimensions:
    dim_1 = K.shape(z_mean)[0]
    dim_2 = K.int_shape(z_mean)[1]
    #sampling:
    norm_sample = K.random_normal(shape=(dim_1, dim_2))
    return z_mean + K.exp(0.5 * z_log_var) * norm_sample


#encoder network:

#regularization coefficient:
l_encode=0.0

#input:
DNAPL_input = Input(shape=(64,40,32,2), name = 'DNAPL')

#CNN layer 1:
x = Conv3D(16, (3, 3, 3), strides=2, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(DNAPL_input)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN layer 2:
x = Conv3D(32, (3, 3, 3), strides=2, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN layer 3:
x = Conv3D(64, (3, 3, 3), strides=2, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN layer 4:
x = Conv3D(128, (3, 3, 3), strides=1, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

x1 = Conv3D(6, (3, 3, 3), strides=1, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(x)
z_mean = Activation('linear')(x1)
shape = K.int_shape(z_mean)
z_mean = Flatten()(z_mean)
x2 = Conv3D(6, (3, 3, 3), strides=1, padding='same',kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_encode))(x)
z_log_var = Activation('linear')(x2)
z_log_var = Flatten()(z_log_var)

#output:
z = Lambda(sampling, output_shape=(960,), name='latent_encode')([z_mean, z_log_var])

#set encoder model:
encoder = Model(DNAPL_input, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

#decoder network:
#regularization coefficient
l_decode=0.0

#input:
latent_input = Input(shape=(960,), name='latent_decode') #input

#reshaping:
x = Reshape((shape[1], shape[2], shape[3], shape[4]))(latent_input)

#CNN/upsampling layer 1:
x = Conv3DTranspose(128, (3, 3, 3), strides=1, padding='same', kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN/upsampling layer 2:
x = Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN/upsampling layer 3:
x = Conv3DTranspose(32, (3, 3, 3), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#CNN/upsampling layer 4:
x = Conv3DTranspose(16, (3, 3, 3), strides=2, padding='same', kernel_initializer=initializers.he_normal(seed=None),kernel_regularizer = regularizers.l2(l_decode))(x)
x = BatchNormalization(axis = 4)(x)
x = Activation('relu')(x)

#output:
outputs1 = Conv3DTranspose(5, (3, 3, 3), activation='softmax', padding='same', name='K')(x)
outputs2 = Conv3DTranspose(10, (3, 3, 3), activation='softmax', padding='same', name='S')(x)
outputs = keras.layers.concatenate([outputs1, outputs2])

#set decoder model:
decoder = Model(latent_input, outputs, name='decoder')
decoder.summary()

#set VAE model
vae_outputs = decoder(encoder(DNAPL_input)[2])
vae = Model(DNAPL_input, vae_outputs, name='vae')

# the starting value of weight is 0
# define it as a keras backend variable
weight = K.variable(1.)
# wrap the loss as a function of weight
def vae_loss(weight):
    def loss (y_true, y_pred):
        l_kl=1.0
        
        y_true_k = K.flatten(y_true[:,:,:,:,0])
        y_pred_k = Reshape((-1,5))(y_pred[:,:,:,:,:5])
        kce_loss = 0.03*81920*sparse_categorical_crossentropy(y_true_k, y_pred_k)

        y_true_s = K.flatten(y_true[:,:,:,:,1])
        y_pred_s = Reshape((-1,10))(y_pred[:,:,:,:,5:])
        sce_loss = 0.03*81920*sparse_categorical_crossentropy(y_true_s, y_pred_s) # need to fine-tune the weight

        kl_loss = - 0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)  #KL loss:
        vae_loss1 = kce_loss+sce_loss+l_kl*weight*K.mean(kl_loss)    #total loss
        return vae_loss1
    return loss

def acc_predK(y_true,y_pred):
   return K.cast(K.equal(   K.cast(y_true[:,:,:,:,0],dtype='int64'), K.argmax(y_pred[:,:,:,:,:5], axis=-1)), K.floatx())

def acc_predS(y_true,y_pred):
   return K.cast(K.equal(   K.cast(y_true[:,:,:,:,1],dtype='int64'), K.argmax(y_pred[:,:,:,:,5:], axis=-1)), K.floatx())

#compile VAE:
Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
vae.compile(optimizer=Adam, loss=vae_loss(weight), metrics=[acc_predK, acc_predS])
vae.summary()

#load/read data
data = hdf5storage.loadmat('KS_evol_55000_64_40_32_2_delSn=0.1_new.mat')
innapl = data['KS'] #velocities
outnapl = data['KS'] #bathymetry

innapl_new=np.reshape(innapl,(-1,64,40,32,2),order='F')
outnapl_new=np.reshape(outnapl,(-1,64,40,32,2),order='F')
print(innapl_new.shape) # size=55000 64 40 32 2
print(outnapl_new.shape)

#add channel dimension:
y_train = outnapl_new #output (Z)
x_train = innapl_new  #input
del innapl_new
del outnapl_new

#check shapes:
print(x_train.shape)
print(y_train.shape)

#normalize input/outputs:
x_train_norm = x_train[:,:,:,:,:]-1
y_train_norm = y_train[:,:,:,:,:]-1

#check shapes:
print(x_train_norm.shape)
print(y_train_norm.shape)

#training:
N=50000 # number of data to be used
val_split=0.1 #validation split
#vae.load_weights('vae_rec.h5') #to load saved weights, if necessary
history=vae.fit(x=x_train_norm[:N,:,:,:,:],y=y_train_norm[:N,:,:,:,:],
        epochs=n_epochs,batch_size=32,shuffle=False,validation_split=val_split,callbacks=[AnnealingCallback(weight)])
vae.save_weights('vae_rec.h5') #to save optimized weights, if necessary

# Plot training & validation accuracy values
plt.plot(history.history['acc_predK'])
plt.plot(history.history['val_acc_predK'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_history_K.png')
plt.close()

# Plot training & validation accuracy values
plt.plot(history.history['acc_predS'])
plt.plot(history.history['val_acc_predS'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy_history_S.png')
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_history.png')
plt.close()

#generate some random samples
z_pred=np.random.randn(100,960)
#predict
y_decoded_pred = decoder.predict(z_pred) #decoder output
#check shape:
print(y_decoded_pred.shape)
#calculate VAE predictions:
y_pred=(y_decoded_pred) #predicted output
savemat('unconditional_sample_100.mat',{'y_pred':y_pred})


#calculate VAE predictions (normalized):
z_mean_pred, z_log_pred, z_pred= encoder.predict(x_train_norm[:,:,:,:,:]) #encoder output
y_decoded_pred = decoder.predict(z_pred) #decoder output

#check shape:
print(y_decoded_pred.shape)

#calculate VAE predictions:
y_pred=y_decoded_pred #predicted output

#check shape:
print(y_pred.shape)

x_train1=x_train[50000:51000,:,:,:,:] # save a small portion of testing set to plot the results;
y_pred1=y_pred[50000:51000,:,:,:,:]
z_mean_pred=z_mean_pred[50000:51000,:]
z_log_pred=z_log_pred[50000:51000,:]
z_pred=z_pred[50000:51000,:]
savemat('reconstructed_realization_testing.mat',{'x_train1':x_train1,'y_pred1':y_pred1,'z_mean_pred':z_mean_pred,'z_log_pred':z_log_pred,'z_pred':z_pred})

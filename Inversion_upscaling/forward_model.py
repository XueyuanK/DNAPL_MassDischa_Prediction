import numpy as np
import os
from keras.models import Model, Sequential, model_from_json
from keras import regularizers
from keras import backend as K
from scipy.io import savemat, loadmat
from keras.losses import mse, binary_crossentropy
from keras.layers import Reshape, Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose
from keras.layers import Activation, ZeroPadding2D, BatchNormalization
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
K.set_image_data_format('channels_last') 
import warnings
warnings.filterwarnings("ignore")


class Model:
    def __init__(self, params=None):

        self.ncores = params['ncores']

        if params is not None:
            self.nx = params['nx']
            self.ny = params['ny']
            self.nz = params['nz']
        else:
            raise ValueError("You have to provide relevant parameters")

    def run_model(self, s):
        '''run forward in comsol
        '''
        # load json and create model
        ############## need to optimize the code here, move the decoder as a parameter read by the func (global var)
        json_file = open('decoder.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        decoder = model_from_json(loaded_model_json)
        # load weights into new model
        decoder.load_weights("decoder.h5")
        KS = decoder.predict(s)  # initial KS images
        Num_ens=KS.shape[0]

        HK = np.argmax(KS[:, :, :, :, :5], axis=-1)+1 # class 1-5, size=Num_ens,64,40,32
        Sn = np.argmax(KS[:, :, :, :, 5:], axis=-1)+1 # class 1-10, size=Num_ens,64,40,32

        HK_obs=HK[:,12:60:12,10:40:10,:] # the 13,25,37,49th nodes; 
        Sn_obs=Sn[:,12:60:12,10:40:10,:]
        HK_obs=np.reshape(HK_obs,(Num_ens,4*3*32),order='F')
        Sn_obs=np.reshape(Sn_obs,(Num_ens,4*3*32),order='F')
        simul_obs=np.hstack((HK_obs,Sn_obs))

        return simul_obs
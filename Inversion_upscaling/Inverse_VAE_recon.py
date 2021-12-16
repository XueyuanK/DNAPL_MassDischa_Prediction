import numpy as np
import os
os.system('module load cuda/9.0')
os.system('module load cudnn/9.0_v7.6.4')
import keras
from keras.models import Model, Sequential,model_from_json
from keras import regularizers
from keras import backend as K
from scipy.io import savemat, loadmat
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.layers import Reshape, Lambda, Input, Dense, Flatten, Conv2D, Conv2DTranspose, Dropout
from keras.layers import Activation, ZeroPadding2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
K.set_image_data_format('channels_last') 
import warnings
warnings.filterwarnings("ignore")
import forward_model
from ES_MDA import ES_MDA
import hdf5storage

# initialization
nx=64
ny=40
nz=32
ncores=1 # for parallel computation
Num_ens=1000
Dim_latent=960
Na=20
Alpha=np.array([20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])

obs=np.loadtxt('obs.txt')
var_obs = np.ones_like(obs)
obs=np.array([obs])
obs=obs.T

# set the error for different kinds of measurements
var_obs[:384]=3e-1 # K
var_obs[384:]=3e-1 # Sn 
R=np.diag(var_obs)

s=np.zeros((Num_ens,Dim_latent,Na+1))
s[:,:,0]=np.random.randn(Num_ens,Dim_latent)

forward_params = params = {'nx': nx, 'ny': ny,'nz': nz,'ncores': ncores}
model = forward_model.Model(forward_params)

for t in range(len(Alpha)):
    sim_obs=model.run_model(s[:,:,t]) # shape of sim_obs (Num_ens,Num_obs)
    np.savetxt('./sim_obs' + str(t) + '.txt', np.mean(sim_obs,axis=0))
    print('RMSE ite_', t, ' : ', np.sqrt(np.mean((np.mean(sim_obs,axis=0)-obs.flatten())**2))) 
    s[:,:,t+1] = ES_MDA(Num_ens, s[:,:,t], obs, sim_obs, Alpha[t], R, [], 2)
sim_obs=model.run_model(s[:,:,len(Alpha)]) # shape of sim_obs (Num_ens,Num_obs)
np.savetxt('./sim_obs' + str(len(Alpha)) + '.txt', np.mean(sim_obs,axis=0))
print('RMSE ite_', len(Alpha), ' : ', np.sqrt(np.mean((np.mean(sim_obs,axis=0)-obs.flatten())**2)))

json_file = open('decoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_model_json)
# load weights into new model
decoder.load_weights("decoder.h5")
# load mean_std K for reconstruction
KS_final_ens = decoder.predict(s[:,:,len(Alpha)])  # KS images for each ens
s_final=np.mean(s[:,:,:],axis=0)    # KS_mean for each step
s_final=s_final.T
KS_mean = decoder.predict(s_final)  # final KS images for the mean s

K_final_ens = np.argmax(KS_final_ens[:, :, :, :, :5], axis=-1)+1 # class 1-5, size=Num_ens,64,40,32
S_final_ens = np.argmax(KS_final_ens[:, :, :, :, 5:], axis=-1)+1 # class 1-10, size=Num_ens,64,40,32
K_mean = np.argmax(KS_mean[:, :, :, :, :5], axis=-1)+1 # class 1-5, size=Num_ens,64,40,32
S_mean = np.argmax(KS_mean[:, :, :, :, 5:], axis=-1)+1 # class 1-5, size=Num_ens,64,40,32

hdf5storage.savemat('results_esmda.mat',{'K_final_ens':K_final_ens,'S_final_ens':S_final_ens,'K_mean':K_mean,'S_mean':S_mean}) 
hdf5storage.savemat('simobs_s.mat',{'s':s,'sim_obs':sim_obs}) # sim_obs for the last step
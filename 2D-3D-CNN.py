'''
2D-3D Convolutional Neural Networks
'''

# To run this code, a listed modules are required
from keras.layers import Input,Conv2D, Conv3D, MaxPooling2D, UpSampling3D, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.backend import tf as ktf


# Parameters
act = 'relu'

# Configurations of input/output variables are as follows:
'''
3D_field: time, nx =256, ny=128, nz=160, components=3
'''

# Ex. 5 cross-sections are used to input of the model
uvw2D_sec[:,:,:,0:3]=uvw3D_field[:,:,:,15,:]
uvw2D_sec[:,:,:,3:6]=uvw3D_field[:,:,:,47,:]
uvw2D_sec[:,:,:,6:9]=uvw3D_field[:,:,:,79,:]
uvw2D_sec[:,:,:,9:12]=uvw3D_field[:,:,:,111,:]
uvw2D_sec[:,:,:,12:]=uvw3D_field[:,:,:,143,:]

# The data is divide to training/validation data
X_train,X_test,y_train,y_test = train_test_split(uvw2D_sec,uvw3D_field,test_size=0.3,random_state=None)


# Input variables
input_field = Input(shape=(256,128,15))

# Network structure
x = Conv2D(32, (3,3),activation=act, padding='same')(input_field)
x = Conv2D(32, (3,3),activation=act, padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(16, (3,3),activation=act, padding='same')(x)
x = Conv2D(20, (3,3),activation=act, padding='same')(x)
x = Reshape([128,64,20,1])(x)
x = Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = Conv3D(16, (3,3,3),activation=act, padding='same')(x)
x = UpSampling3D((2,2,8))(x)
x = Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x = Conv3D(32, (3,3,3),activation=act, padding='same')(x)
x_final = Conv3D(3,(3,3,3),activation='linear', padding='same')(x)
# -------- #
model = Model(input_field,x_final)
model.compile(optimizer='adam',loss='mse')
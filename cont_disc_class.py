# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:20:44 2019

@author: Xu jingyu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.losses import binary_crossentropy   
from tensorflow.keras.layers import Lambda, Input, Dense,Concatenate,Dropout,PReLU 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras import metrics
from keras import regularizers
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sklearn
from keras.utils import plot_model
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral,denoise_wavelet, estimate_sigma
#import operator
#import collections
#import scipy.io as sio

original_dim=4096
latent_cont_dim=5
latent_disc_dim=7
latent_dim=latent_cont_dim+latent_disc_dim
epsilon_std=0.3
EPSILON = 1e-8
epochs = 1
#==============================================================================
# train = sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\train.mat')
# train_label=sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\train_label.mat')
# test=sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\test.mat')
# test_label = sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\test_label.mat')
#==============================================================================
x_train=train17
y_train=train_label17
x_test=test15
y_test=test_label15

x_zero = T72.astype('float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_zero = T72_label.astype('float32')
y_train=y_train.astype('float32')
y_test=y_test.astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_zero = x_zero.reshape((len(x_zero),np.prod(x_zero.shape[1:])))
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)) 
x_train = x_train.transpose(1,0)
x_test = x_test.transpose(1,0)
x_zero = x_zero.transpose(1,0)                    
x_train = scaler.fit_transform(x_train)      
x_test = scaler.fit_transform(x_test)     
x_zero = scaler.fit_transform(x_zero)  
x_train = x_train.transpose(1,0)
x_test = x_test.transpose(1,0)           
x_zero = x_zero.transpose(1,0)
       
def get_one_hot_vector(idx, dim):
    one_hot = np.zeros(dim)
    one_hot[idx] = 1.
    return one_hot                       

def sampling_normal(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch,dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(0.5*z_log_sigma) * epsilon

def sampling_concrete(alpha,temperature=0.67):
    batch=K.shape(alpha)[0]
    dim=K.int_shape(alpha)[1]
    uniform = K.random_uniform(shape=(batch,dim))
    gumbel = - K.log(- K.log(uniform + EPSILON) + EPSILON)
    fenshi=tf.reduce_logsumexp((K.log(alpha+EPSILON)+gumbel)/temperature,1,keepdims=True)
    logit = (K.log(alpha + EPSILON) + gumbel) / temperature -fenshi
    return logit
 #   return K.softmax(logit)

def kl_discrete(dist,y,yk,temperature=0.67):
    dim=K.shape(dist)[1]
    dim=tf.cast(dim,tf.float32)
    kl_batch1=K.sum(K.log(dist+EPSILON)-temperature*yk,axis=1)-K.sum(K.log(y+EPSILON)-temperature*yk,axis=1)
    kl_batch2=-dim*tf.reduce_logsumexp(K.log(dist+EPSILON)-temperature*yk,1)+dim*tf.reduce_logsumexp(K.log(y+EPSILON)-temperature*yk,1)
    return tf.reduce_sum(kl_batch1+kl_batch2)

def mse(imageA, imageB):
    err = tf.squared_difference(imageA , imageB )
    err = tf.reduce_sum(err)
    #err /= original_dim
    return err

x=Input(batch_shape=(None,original_dim))
y=Input(batch_shape=(None,latent_disc_dim))

a0=Dense(1024,activation='relu',
         kernel_regularizer=regularizers.l2(0.01))(x)
#a0=Dropout(0.5)(a0)

a1 = Dense(512, activation='relu',
           kernel_regularizer=regularizers.l2(0.01))(a0)
#a1 = Dropout(0.5)(a1)

a2 = Dense(128,activation='relu',
           kernel_regularizer=regularizers.l2(0.01))(a1)

a3=Dense(64, activation='relu',
         kernel_regularizer=regularizers.l2(0.01))(a2)
#a3=Dropout(0.5)(a3)

z_mean = Dense(latent_cont_dim, name='cont_layer',
               kernel_regularizer=regularizers.l2(0.01))(a3)

z_log_sigma = Dense(latent_cont_dim,
                    kernel_regularizer=regularizers.l2(0.01))(a3)

alpha= Dense(latent_disc_dim, activation='softmax',name='disc_layer',
             kernel_regularizer=regularizers.l2(0.01))(a3)

encoder = Model([x,y],[z_mean,alpha])

z = Lambda(sampling_normal)([z_mean, z_log_sigma])
c = Lambda(sampling_concrete)(alpha)
encoding = Concatenate()([z, c])
# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
# end-to-end autoencoder
decoder_mean1=Dense(64,activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))

decoder_mean2 = Dense(128,activation='relu',
                      kernel_regularizer=regularizers.l2(0.01))

decoder_mean3=Dense(512,activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))

decoder_mean4=Dense(1024,activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))

decoder_mean5=Dense(original_dim,activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))
       
decoded1=decoder_mean1(encoding)   
#decoded1=Dropout(0.5)(decoded1)
decoded2=decoder_mean2(decoded1)
#decoded2=Dropout(0.5)(decoded2)
decoded3=decoder_mean3(decoded2)
#decoded3=Dropout(0.5)(decoded3)
decoded4=decoder_mean4(decoded3)
#decoded4=Dropout(0.5)(decoded4)
x_decoded_mean=decoder_mean5(decoded4)
vae=Model([x,y], x_decoded_mean)

generator1 = Input(shape=(latent_dim,))
generator2=decoder_mean1(generator1)
generator3=decoder_mean2(generator2)
generator4=decoder_mean3(generator3)
generator5=decoder_mean4(generator4)
generator6=decoder_mean5(generator5)

generator = Model(generator1, generator6)

mse_loss = mse(x,x_decoded_mean)
label_loss = binary_crossentropy(y , alpha)
kl_normal_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
kl_disc_loss = kl_discrete(alpha,y,c)

def vae_loss(y,alpha):
    vae_loss=mse_loss + label_loss + 2*kl_normal_loss + kl_disc_loss
    return vae_loss
 
vae.compile(optimizer='adam', loss=vae_loss)

vae.fit([x_train,y_train],x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=64,
        validation_data=([x_test,y_test],x_test))

cont_layer_model=Model(inputs=encoder.input,outputs=encoder.get_layer('cont_layer').output)

latent_cont_data=cont_layer_model.predict([x_test,y_test])
T72_cont_pred = cont_layer_model.predict([x_zero,y_zero])

zS1_mean=np.mean(latent_cont_data[0:273],0)
BRDM2_mean=np.mean(latent_cont_data[274::547],0)
BTR60_mean=np.mean(latent_cont_data[548:742],0)
D7_mean=np.mean(latent_cont_data[743:1016],0)
T62_mean=np.mean(latent_cont_data[1017:1289],0)
ZIL131_mean=np.mean(latent_cont_data[1290:1563],0)
ZSU234_mean=np.mean(latent_cont_data[1564:1837],0)
test_mean=np.vstack((zS1_mean,BRDM2_mean,BTR60_mean,D7_mean,T62_mean,ZIL131_mean,ZSU234_mean))

zS1=0
BRDM2=0
BTR60=0
D7=0
T62=0
ZIL131=0
ZSU234=0
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(test_mean)
distances, indices = nbrs.kneighbors(T72_cont_pred)
for m in range(len(indices)):
    if(indices[m]==0):zS1+=1
    elif(indices[m]==1):BRDM2+=1
    elif(indices[m]==2):BTR60+=1
    elif(indices[m]==3):D7+=1
    elif(indices[m]==4):T62+=1
    elif(indices[m]==5):ZIL131+=1
    else: ZSU234+=1
print('zS1_cont_class:',zS1)
print('BRDM2_cont_class:',BRDM2)
print('BTR60_cont_class:',BTR60)
print('D7_cont_class:',D7)
print('T62_cont_class:',T62)
print('ZIL131_cont_class:',ZIL131)
print('ZSU234_cont_class:',ZSU234)
print('\n'*2) 

         
zS11=0
BRDM21=0
BTR601=0
D71=0
T621=0
ZIL1311=0
ZSU2341=0
disc_layer_model=Model(inputs=encoder.input,outputs=encoder.get_layer('disc_layer').output)
latent_disc_pred=disc_layer_model.predict([x_zero,y_zero])
#disc_mean=np.mean(latent_disc_pred,0)
predictions = tf.argmax(latent_disc_pred, 1)
predictions=tf.Session().run(predictions)
for n in range(len(predictions)):
    if(predictions[n]==0):zS11+=1
    elif(predictions[n]==1):BRDM21+=1
    elif(predictions[n]==2):BTR601+=1
    elif(predictions[n]==3):D71+=1
    elif(predictions[n]==4):T621+=1
    elif(predictions[n]==5):ZIL1311+=1
    elif(predictions[n]==6):ZSU2341+=1
print('zS1_disc_class:',zS11)
print('BRDM2_disc_class:',BRDM21)
print('BTR60_disc_class:',BTR601)
print('D7_disc_class:',D71)
print('T62_disc_class:',T621)
print('ZIL131_disc_class:',ZIL1311)
print('ZSU234_disc_class:',ZSU2341)
print('\n'*2)    

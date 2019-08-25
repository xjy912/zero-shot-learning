# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 14:13:12 2019

@author: xu jingyu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Lambda, Input, Dense,Concatenate,Conv2D, MaxPooling2D, UpSampling2D,Flatten,Dropout,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras import metrics
from keras import regularizers
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import operator
import collections
import cv2
import sklearn

original_dim=4096
latent_cont_dim=5
latent_disc_dim=7
latent_dim=latent_cont_dim+latent_disc_dim
epochs=10
epsilon_std=0.3
EPSILON = 1e-8

#train = sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\train.mat')
# train_label=sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\train_label.mat')
# test=sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\test.mat')
# test_label = sio.loadmat(r'C:\Users\xujingyu\Desktop\少样本\test_label.mat')
x_train=train17
y_train=train_label17
x_test=test15
y_test=test_label15

x_zero = T72.astype('float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_zero = T72_label.astype('float32')
y_train=y_train.astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
x_zero = x_zero.reshape((len(x_zero),np.prod(x_zero.shape[1:])))
scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))               
x_train = scaler.fit_transform(x_train)      
x_test = scaler.fit_transform(x_test)     
x_zero = scaler.fit_transform(x_zero)  

x_train = x_train.reshape(len(x_train),64,64,1)
x_test = x_test.reshape(len(x_test),64,64,1)
x_zero = x_zero.reshape(len(x_zero),64,64,1)
y_test=y_test.astype('float32')

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
    fenshi=tf.reduce_logsumexp((K.log(alpha+EPSILON)+gumbel)/temperature,1,keep_dims=True)
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

#==============================================================================
# def flatten1(a):
#     x_flat = tf.reshape(a,[K.shape(a)[0], np.prod(a.shape[1:])])
#     return x_flat
#==============================================================================
def Squeeze(a):
    n = tf.squeeze(a,[1,2])
    return n

def replace(b):
    encoded = tf.reshape(b,[K.shape(b)[0],1,1,12])
    return encoded

x = Input(shape=( 64 , 64,1))
y = Input(batch_shape=(None,latent_disc_dim))
x1 = Conv2D(32, (3,3),2, activation='relu', padding ='same')(x)
#x2 = MaxPooling2D((2, 2), padding='same')(x1)
x2 = Conv2D(64, (3,3),2, activation='relu', padding ='same')(x1)
#x4 = MaxPooling2D((2, 2), padding='same')(x3)
x3 = Conv2D(64, (3,3),2, activation='relu', padding ='same')(x2)
x4 = Conv2D(128,(3,3),2, activation='relu',padding='same')(x3)
x5 = Conv2D(128,(3,3),2, activation='relu',padding='same')(x4)
#x5 = Dropout(0.5)(x5)

alpha = Conv2D(latent_disc_dim,(2,2),activation='softmax',name='disc_layer')(x5)
alpha1 = Lambda(Squeeze)(alpha)

z_mean = Conv2D(latent_cont_dim, (2,2),name='cont_layer')(x5)
z_mean1= Lambda(Squeeze)(z_mean)

z_log_sigma = Conv2D(latent_cont_dim, (2,2))(x5)
z_log_sigma1= Lambda(Squeeze)(z_log_sigma)

encoder = Model([x,y],[z_mean,alpha])

z = Lambda(sampling_normal)([z_mean1, z_log_sigma1])
c = Lambda(sampling_concrete)(alpha1)
encoding = Concatenate()([z, c])
encoding1 = Lambda(replace)(encoding)
#encoding1 = Dropout(0.5)(encoding1)

decoded1 = Conv2DTranspose(128,(3,3), 2,activation='relu',padding='same')(encoding1)
decoded2 = Conv2DTranspose(128,(3,3), 2,activation='relu',padding='same')(decoded1)
decoded3 = Conv2DTranspose(64,(3,3), 2,activation='relu',padding='same')(decoded2)
decoded4 = Conv2DTranspose(64,(3,3), 2,activation='relu',padding='same')(decoded3)
decoded5 = Conv2DTranspose(32,(3,3), 2,activation='relu',padding='same')(decoded4)
x_decoded_mean = Conv2DTranspose(1, (3, 3), 2,activation='relu', padding ='same')(decoded5)
vae=Model([x,y], x_decoded_mean)

encoding1 = Input(shape=(1,1,12))
decoded1 = Conv2DTranspose(128,(3,3), 2,activation='relu',padding='same')(encoding1)
decoded2 = Conv2DTranspose(128,(3,3), 2,activation='relu',padding='same')(decoded1)
decoded3 = Conv2DTranspose(64,(3,3), 2,activation='relu',padding='same')(decoded2)
decoded4 = Conv2DTranspose(64,(3,3), 2,activation='relu',padding='same')(decoded3)
decoded5 = Conv2DTranspose(32,(3,3), 2,activation='relu',padding='same')(decoded4)
x_decoded_mean = Conv2DTranspose(1, (3, 3), 2,activation='relu', padding ='same')(decoded5)
generator=Model(encoding1,x_decoded_mean)

def vae_loss(x,x_decoded_mean):
    mse_loss = mse(x,x_decoded_mean)
    xent_loss = binary_crossentropy(y , alpha1)
    kl_normal_loss = - 0.5 * K.mean(1 + z_log_sigma1 - K.square(z_mean1) - K.exp(z_log_sigma1), axis=-1)
    kl_disc_loss = kl_discrete(alpha1,y,c)
    return  mse_loss + xent_loss*EPSILON + kl_normal_loss*EPSILON + kl_disc_loss*EPSILON
 
vae.compile(optimizer='adam', loss=vae_loss)

#encoded = MaxPooling2D((2, 2), padding='same')(x_last)
#x_flat = Lambda (flatten1)(encoded)
#x_flat = Flatten()(encoded)
#x6 = UpSampling2D((2, 2))(x5)
for n in range(60):
    histrory=vae.fit([x_train,y_train],x_train,
                      shuffle=True,
                      epochs=epochs,
                      batch_size=64,
                      validation_data=([x_test,y_test],x_test))
    my_layer_model=Model(inputs=encoder.input,outputs=encoder.get_layer('disc_layer').output)
    y_disc_pred=my_layer_model.predict([x_test,y_test])
    y_disc_pred1=Squeeze(y_disc_pred)
    predictions = tf.argmax(y_disc_pred1, 1)
    actuals = tf.argmax(y_test,1)
    score=tf.reduce_sum(
          tf.cast(tf.equal(predictions,actuals),"float32"))
    accuracy=tf.reduce_mean(
          tf.cast(tf.equal(predictions,actuals),"float32"))
    score=tf.Session().run(score)
    accuracy=tf.Session().run(accuracy)
    print('score:',score)
    print('accuracy:',accuracy)
    
    n = 10 # figure with 15x15 digits
    digit_size = 64
    figure = np.zeros((digit_size*n,digit_size*n))
    grid_x = np.linspace(-10,10,n)
    for j in range(latent_cont_dim):
        for i,xi in enumerate(grid_x):
            z_sample =get_one_hot_vector(j,latent_cont_dim)*xi
            c_sample =np.array([1,0,0,0,0,0,0])
            latent_sample = np.hstack((z_sample, c_sample))
            latent_sample=latent_sample.reshape(1,1,1,12)
            x_decoded = generator.predict(latent_sample)
            digit = x_decoded[0].reshape(digit_size,digit_size)
            figure[j*digit_size:(j+1)*digit_size,
                   i* digit_size: (i + 1) * digit_size] = digit
    cv2.imshow('image',figure)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()

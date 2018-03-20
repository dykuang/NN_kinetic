# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:30:30 2018

@author: dykuang

Building a network to predict kinetic triplets.
"""
from keras.models import Model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from keras.layers import Conv1D, Input, GaussianNoise, Flatten, Dropout, Dense,\
                        BatchNormalization, MaxPooling1D, concatenate, add,\
                        GlobalAveragePooling1D

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------
cls = 11
batchsize = 256
epochs = 250
preprocess = True

xTrain = np.load(r'dataset/xtrain.npy')
yTrain = np.load(r'dataset/ytrain.npy')

yTrain_label = yTrain[:,0]
yTrain_para = yTrain[:,1:]

yTrain_label = utils.to_categorical(yTrain_label, cls)
#------------------------------------------------------------------------------
# preprocess
#------------------------------------------------------------------------------
if preprocess:
    Scaler = MinMaxScaler().fit(xTrain)
    x_train_std = Scaler.transform(xTrain)
else:
    x_train_std = xTrain
    
    
    
#------------------------------------------------------------------------------
# build network
#------------------------------------------------------------------------------
#from keras.layers.advanced_activations import PReLU
input_dim = x_train_std.shape[1]
feature = Input(shape = (input_dim, 1))
#
x = GaussianNoise(0.01)(feature)
x = Conv1D(filters= 16, kernel_size = 4, strides=4, padding='valid',  
       activation='relu',name = 'conv1D_1')(x)
x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
x = Flatten(name = 'flat_1')(x)

x_x = GaussianNoise(0.01)(feature)
x_x = Conv1D(filters= 24, kernel_size = 6, strides= 6, padding='valid',
       activation='relu',name = 'conv1D_2')(x_x)
x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_2')(x_x)
x_x = Flatten(name = 'flat_2')(x_x)

x_x_x = GaussianNoise(0.01)(feature)
x_x_x = Conv1D(filters= 32, kernel_size = 8, strides= 8, padding='valid', 
       activation='relu', name = 'conv1D_3')(x_x_x)
x_x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_3')(x_x_x)
x_x_x = Flatten(name = 'flat_3')(x_x_x)


feature_f = GaussianNoise(0.01)(feature)
feature_f = Flatten(name = 'flat_4')(feature_f)
##
x = concatenate([x, x_x, x_x_x, feature_f])
#x = Dense(128, activation = 'relu', name = 'dense_1')(x)

#x = Conv1D(filters= 32, kernel_size = 3, strides=3, padding='same',  
#           activation='relu',name = 'conv1D_1')(feature)
#x = Conv1D(filters= 64, kernel_size = 3, strides=3, padding='same',  
#           activation='relu',name = 'conv1D_2')(x)
#x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)

#x = Flatten()(x)
#x = add([feature, x])
#x = BatchNormalization()(x)
#x = GlobalAveragePooling1D()(x)

x = Dense(256, activation = 'relu', name = 'dense_0')(x)
#x = Dense(128, activation = 'relu', name = 'dense')(x)
x1 = Dense(64, activation = 'relu', name = 'dense_1')(x)
x2 = Dense(32, activation = 'relu', name = 'dense_2')(x)

pred = Dense(cls, activation = 'softmax', name = 'which_model')(x1)
par = Dense(2, activation = 'relu', name = 'ElnA')(x2)

model = Model(feature, [pred, par])
#model = Model(feature, pred)
#best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#best_model = ModelCheckpoint(target_dir+'leaf_conv1d.hdf5', monitor='val_acc', 
#                             verbose=1, save_best_only=True, save_weights_only=False, 
#                             mode='auto', period=1)

#def customloss(yTrue, yPred):
#    cls_loss = losses.categorical_crossentropy(yTrue[0], yPred[0])
#    reg_loss = losses.mean_squared_logarithmic_error(yTrue[1], yPred[1])
#    return reg_loss 
    
model.compile(loss ={'which_model': 'categorical_crossentropy', 'ElnA': 'mean_squared_logarithmic_error'},
              loss_weights={'which_model': 5.0, 'ElnA': 0.0},
              optimizer = 'adam',
#            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
            metrics = {'which_model': 'accuracy'}
            )

#model.compile(loss = 'categorical_crossentropy',
#              optimizer = 'adam')
#x_train_std = x_train_std.reshape(x_train_std.shape[0], x_train_std.shape[1], 1)
#x_test_std = x_test_std.reshape(x_test_std.shape[0], x_test_std.shape[1], 1)
x_train_std = np.expand_dims(x_train_std, 2)

#history = model.fit(x=x_train_std, y= yTrain_label,
#                    batch_size = batchsize,
#                    epochs = epochs, verbose = 0)
#
history = model.fit(x=x_train_std, y= [yTrain_label, yTrain_para],
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
#                    validation_split = .2,
#                    validation_data = (x_test_std, y_test),
#                    callbacks=[best_model]
                    )

plt.subplot(1,2,1)
plt.plot(history.history['which_model_loss'])
plt.subplot(1,2,2)
plt.plot(history.history['ElnA_loss'])

score = model.evaluate(x_train_std, [yTrain_label, yTrain_para])

print("The accuracy is : {}".format(score[3]))
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:30:30 2018

@author: dykuang

Building a network to predict kinetic triplets.

It turns out an easier structure does much better than the more complicated structure.

(should start with easy ones)

best accuracy: 82%

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
batchsize = 128
epochs = 150
preprocess = True

#xTrain = np.load(r'dataset/xtrain1.npy')
#xTrain = np.hstack([xTrain, np.diff(xTrain, axis = 1)])
#yTrain = np.load(r'dataset/ytrain1.npy')

x_Train1 = np.load(r'dataset/xtrain_sep1.npy')
x_Train2 = np.load(r'dataset/xtrain_sep2.npy')
x_Train3 = np.load(r'dataset/xtrain_sep3.npy')

xTrain = np.hstack([x_Train1, x_Train2, x_Train3])
xTrain = np.hstack([xTrain, np.diff(xTrain, axis = 1)])
yTrain = np.load(r'dataset/ytrain_sep.npy')

#yTrain_label = yTrain[:,0]
#yTrain_para = yTrain[:,1:]



#------------------------------------------------------------------------------
# Train/Test split
#------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, 
                                                    test_size = 0.3,
                                                    random_state = 123,
                                                    stratify = yTrain[:,0])
y_train_label = utils.to_categorical(y_train[:,0], cls)
y_test_label = utils.to_categorical(y_test[:, 0], cls)
#------------------------------------------------------------------------------
# preprocess
#------------------------------------------------------------------------------
if preprocess:
    Scaler = StandardScaler().fit(x_train)
    x_train_std = Scaler.transform(x_train)
    x_test_std = Scaler.transform(x_test)
else:
    x_train_std = x_train
    
    
    
#------------------------------------------------------------------------------
# build network
#------------------------------------------------------------------------------
#from keras.layers.advanced_activations import PReLU
input_dim = x_train_std.shape[1]
feature = Input(shape = (input_dim, 1))
#
# =============================================================================
# x = GaussianNoise(0.01)(feature)
# x = Conv1D(filters= 8, kernel_size = 4, strides=4, padding='valid',  
#        activation='relu',name = 'conv1D_1')(x)
# x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
# x = Flatten(name = 'flat_1')(x)
# 
# x_x = GaussianNoise(0.01)(feature)
# x_x = Conv1D(filters= 12, kernel_size = 6, strides= 6, padding='valid',
#        activation='relu',name = 'conv1D_2')(x_x)
# x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_2')(x_x)
# x_x = Flatten(name = 'flat_2')(x_x)
# 
# x_x_x = GaussianNoise(0.01)(feature)
# x_x_x = Conv1D(filters= 16, kernel_size = 8, strides= 8, padding='valid', 
#        activation='relu', name = 'conv1D_3')(x_x_x)
# x_x_x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_3')(x_x_x)
# x_x_x = Flatten(name = 'flat_3')(x_x_x)
# 
# 
# feature_f = GaussianNoise(0.01)(feature)
# feature_f = Flatten(name = 'flat_4')(feature_f)
# ##
# x = concatenate([x, x_x])
# =============================================================================
#x = Dense(128, activation = 'relu', name = 'dense_1')(x)

x = Conv1D(filters= 32, kernel_size = 3, strides=3, padding='same',  
           activation='relu',name = 'conv1D_1')(feature)
#x = Conv1D(filters= 64, kernel_size = 3, strides=3, padding='same',  
#           activation='relu',name = 'conv1D_2')(x)
x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)

x = Flatten()(x)
#x = add([feature, x])
#x = BatchNormalization()(x)
#x = GlobalAveragePooling1D()(x)

x = Dense(64, activation = 'relu', name = 'dense_0')(x)
x = Dropout(0.25)(x)
#x = BatchNormalization()(x)
#x = Dense(128, activation = 'relu', name = 'dense')(feature)
x1 = Dense(32, activation = 'relu', name = 'dense_1')(x)
#x2 = Dense(16, activation = 'relu', name = 'dense_2')(x)

x1 = Dropout(0.25)(x1)
pred = Dense(cls, activation = 'softmax', name = 'which_model')(x1)
par = Dense(2, activation = 'relu', name = 'ElnA')(x1)

model = Model(feature, [pred, par])

#best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#best_model = ModelCheckpoint(target_dir+'leaf_conv1d.hdf5', monitor='val_acc', 
#                             verbose=1, save_best_only=True, save_weights_only=False, 
#                             mode='auto', period=1)

#def customloss(yTrue, yPred):
#    cls_loss = losses.categorical_crossentropy(yTrue[0], yPred[0])
#    reg_loss = losses.mean_squared_logarithmic_error(yTrue[1], yPred[1])
#    return reg_loss 
    
model.compile(loss ={'which_model': 'categorical_crossentropy', 
                     'ElnA': 'mean_squared_logarithmic_error'},
              loss_weights={'which_model': 1.0, 'ElnA': 3.0},
              optimizer = 'adam',
#            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
            metrics = {'which_model': 'accuracy'}
            )


x_train_std = np.expand_dims(x_train_std, 2)
x_test_std = np.expand_dims(x_test_std, 2)
#history = model.fit(x=x_train_std, y= yTrain_label,
#                    batch_size = batchsize,
#                    epochs = epochs, verbose = 0)
#
history = model.fit(x=x_train_std, y= [y_train_label, y_train[:,1:]],
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
#                    validation_split = .2,
                    validation_data = (x_test_std, [y_test_label, y_test[:,1:]])
#                    callbacks=[best_model]
                    )

plt.subplot(1,2,1)
plt.plot(history.history['which_model_loss'])
plt.plot(history.history['val_which_model_loss'])
plt.subplot(1,2,2)
plt.plot(history.history['ElnA_loss'])
plt.plot(history.history['val_ElnA_loss'])

score = model.evaluate(x_test_std, [y_test_label, y_test[:,1:]])

print("The accuracy is : {}".format(score[3]))
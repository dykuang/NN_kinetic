# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 00:26:51 2018

@author: dykuang

cross validation for accessing the model performance
"""
from keras.models import Model
from keras import backend as K
from keras import optimizers, losses, utils
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
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
epochs = 200
preprocess = True


x_Train1 = np.load(r'dataset/xtrain_sep1.npy')
x_Train2 = np.load(r'dataset/xtrain_sep2.npy')
x_Train3 = np.load(r'dataset/xtrain_sep3.npy')

xTrain = np.hstack([x_Train1, x_Train2, x_Train3])
#xTrain = np.hstack([xTrain, np.diff(xTrain, axis = 1)])
yTrain = np.load(r'dataset/ytrain_sep.npy')

yTrain[:,1] = yTrain[:,1]/10

from scipy import io
pine_test1 = io.loadmat(r'dataset/pine5.mat')['aT'].transpose()
pine_test2 = io.loadmat(r'dataset/pine10.mat')['aT'].transpose()
pine_test3 = io.loadmat(r'dataset/pine15.mat')['aT'].transpose()

test_pine = np.hstack([pine_test1, pine_test2, pine_test3])
#test_pine = np.hstack([test_pine, np.diff(test_pine, axis = 1)])

corn_test1 = io.loadmat(r'dataset/corn5.mat')['aT'].transpose()
corn_test2 = io.loadmat(r'dataset/corn10.mat')['aT'].transpose()
corn_test3 = io.loadmat(r'dataset/corn15.mat')['aT'].transpose()

test_corn = np.hstack([corn_test1, corn_test2, corn_test3])
#test_corn = np.hstack([test_corn, np.diff(test_corn, axis = 1)])

coal_test1 = io.loadmat(r'dataset/coal5.mat')['aT'].transpose()
coal_test2 = io.loadmat(r'dataset/coal10.mat')['aT'].transpose()
coal_test3 = io.loadmat(r'dataset/coal15.mat')['aT'].transpose()

test_coal = np.hstack([coal_test1, coal_test2, coal_test3])
#test_coal = np.hstack([test_coal, np.diff(test_coal, axis = 1)])


score = []
pred_pine=[]
pred_corn=[]
pred_coal=[]
#------------------------------------------------------------------------------
# 5-fold cross-validation
#------------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
for train_index, test_index in skf.split(xTrain, yTrain[:,0]):

    x_train, x_test = xTrain[train_index], xTrain[test_index]
    y_train, y_test = yTrain[train_index], yTrain[test_index]
    y_train_label = utils.to_categorical(y_train[:,0], cls)
    y_test_label = utils.to_categorical(y_test[:, 0], cls)


    #------------------------------------------------------------------------------
    # preprocess
    #------------------------------------------------------------------------------
    Scaler = StandardScaler().fit(x_train)
    x_train_std = Scaler.transform(x_train)
    x_test_std = Scaler.transform(x_test)
#    test_pine_std = Scaler.transform(test_pine)
#    test_corn_std = Scaler.transform(test_corn)
#    test_coal_std = Scaler.transform(test_coal)

    
    x_train_std = np.expand_dims(x_train_std, 2)
    x_test_std = np.expand_dims(x_test_std, 2)
#    test_pine_std = np.expand_dims(test_pine_std, 2)
#    test_corn_std = np.expand_dims(test_corn_std, 2)
#    test_coal_std = np.expand_dims(test_coal_std, 2)
    
    #------------------------------------------------------------------------------
    # build network
    #------------------------------------------------------------------------------
    input_dim = x_train_std.shape[1]
    feature = Input(shape = (input_dim, 1))
    x = GaussianNoise(0.1)(feature)
    
    x = Conv1D(filters= 4, kernel_size = 3, strides=3, padding='valid',  
               activation='relu',name = 'conv1D_1')(feature)
    
    x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)
    
    x = Flatten()(x)
    
    
    x = Dense(64, activation = 'relu', name = 'dense_0')(x)
    
    x1 = Dense(32, activation = 'relu', name = 'dense_1')(x)
    
#    x1 = Dropout(0.2)(x1)
    pred = Dense(cls, activation = 'softmax', name = 'which_model')(x1)
    
    
    par1 = Dense(1, activation = 'relu', name = 'E')(x1)
    par2 = Dense(1, activation = 'relu', name = 'lnA')(x1)
    model = Model(feature, [pred, par1, par2])
    
        
    model.compile(loss ={'which_model': 'categorical_crossentropy', 
                         'E': 'mean_absolute_percentage_error',
                         'lnA': 'mean_absolute_percentage_error'},
                  loss_weights={'which_model': 10.0, 'E': 1.0, 'lnA': 1.0},
                  optimizer = 'adam',
    #            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
                metrics = {'which_model': 'accuracy'}
                )
#
    history = model.fit(x=x_train_std, y= [y_train_label, y_train[:,1], y_train[:,2]],
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0
                    )


    score.append(model.evaluate(x_test_std, [y_test_label, y_test[:,1], y_test[:,2]]))
#    pred_pine.append(model.predict(test_pine_std))
#    pred_corn.append(model.predict(test_corn_std))
#    pred_coal.append(model.predict(test_coal_std))


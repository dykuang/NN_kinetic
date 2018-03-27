# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:30:30 2018

@author: dykuang

Building a network to predict kinetic triplets.

It turns out an easier structure does much better than the more complicated structure.

(should start with easy ones)

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
epochs = 160
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

yTrain[:,1] = yTrain[:,1]/10
#yTrain[:,1] = yTrain[:,2]/10
#yTrain_label = yTrain[:,0]
#yTrain_para = yTrain[:,1:]

from scipy import io
pine_test1 = io.loadmat(r'dataset/pine5.mat')['aT'].transpose()
pine_test2 = io.loadmat(r'dataset/pine10.mat')['aT'].transpose()
pine_test3 = io.loadmat(r'dataset/pine15.mat')['aT'].transpose()

test_pine = np.hstack([pine_test1, pine_test2, pine_test3])
test_pine = np.hstack([test_pine, np.diff(test_pine, axis = 1)])

corn_test1 = io.loadmat(r'dataset/corn5.mat')['aT'].transpose()
corn_test2 = io.loadmat(r'dataset/corn10.mat')['aT'].transpose()
corn_test3 = io.loadmat(r'dataset/corn15.mat')['aT'].transpose()

test_corn = np.hstack([corn_test1, corn_test2, corn_test3])
test_corn = np.hstack([test_corn, np.diff(test_corn, axis = 1)])

coal_test1 = io.loadmat(r'dataset/coal5.mat')['aT'].transpose()
coal_test2 = io.loadmat(r'dataset/coal10.mat')['aT'].transpose()
coal_test3 = io.loadmat(r'dataset/coal15.mat')['aT'].transpose()

test_coal = np.hstack([coal_test1, coal_test2, coal_test3])
test_coal = np.hstack([test_coal, np.diff(test_coal, axis = 1)])

#------------------------------------------------------------------------------
# Train/Test split
#------------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(xTrain, yTrain, 
                                                    test_size = 0.25,
                                                    random_state = 1234,
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
    test_pine = Scaler.transform(test_pine)
    test_corn = Scaler.transform(test_corn)
    test_coal = Scaler.transform(test_coal)
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

x = Conv1D(filters= 4, kernel_size = 3, strides=3, padding='valid',  
           activation='relu',name = 'conv1D_1')(feature)
#x = Conv1D(filters= 64, kernel_size = 3, strides=3, padding='same',  
#           activation='relu',name = 'conv1D_2')(x)
x = MaxPooling1D(pool_size=2, strides=2, name = 'MP_1')(x)

x = Flatten()(x)
#x = add([feature, x])
#x = BatchNormalization()(x)
#x = GlobalAveragePooling1D()(x)

x = Dense(64, activation = 'relu', name = 'dense_0')(x)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
#x = Dense(128, activation = 'relu', name = 'dense')(feature)
x1 = Dense(32, activation = 'relu', name = 'dense_1')(x)
#x2 = Dense(16, activation = 'relu', name = 'dense_2')(x)

x1 = Dropout(0.2)(x1)
pred = Dense(cls, activation = 'softmax', name = 'which_model')(x1)

#par = Dense(2, activation = 'relu', name = 'ElnA')(x1)
par1 = Dense(1, activation = 'relu', name = 'E')(x1)
par2 = Dense(1, activation = 'relu', name = 'lnA')(x1)
model = Model(feature, [pred, par1, par2])

#best_model=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
#best_model = ModelCheckpoint(target_dir+'leaf_conv1d.hdf5', monitor='val_acc', 
#                             verbose=1, save_best_only=True, save_weights_only=False, 
#                             mode='auto', period=1)

#def customloss(yTrue, yPred):
#    cls_loss = losses.categorical_crossentropy(yTrue[0], yPred[0])
#    reg_loss = losses.mean_squared_logarithmic_error(yTrue[1], yPred[1])
#    return reg_loss 
    
model.compile(loss ={'which_model': 'categorical_crossentropy', 
                     'E': 'mean_squared_logarithmic_error',
                     'lnA': 'mean_squared_logarithmic_error'},
              loss_weights={'which_model': 1.0, 'E': 2.0, 'lnA': 2.0},
              optimizer = 'adam',
#            optimizer = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True),
            metrics = {'which_model': 'accuracy'}
            )


x_train_std = np.expand_dims(x_train_std, 2)
x_test_std = np.expand_dims(x_test_std, 2)
test_pine = np.expand_dims(test_pine, 2)
test_corn = np.expand_dims(test_corn, 2)
test_coal = np.expand_dims(test_coal, 2)
#history = model.fit(x=x_train_std, y= yTrain_label,
#                    batch_size = batchsize,
#                    epochs = epochs, verbose = 0)
#
history = model.fit(x=x_train_std, y= [y_train_label, y_train[:,1], y_train[:,2]],
                    batch_size = batchsize,
                    epochs = epochs, verbose = 0,
#                    validation_split = .2,
                    validation_data = (x_test_std, [y_test_label, y_test[:,1], y_test[:,2]])
#                    callbacks=[best_model]
                    )

plt.subplot(1,3,1)
plt.plot(history.history['which_model_loss'])
plt.plot(history.history['val_which_model_loss'])
plt.subplot(1,3,2)
plt.plot(history.history['E_loss'])
plt.plot(history.history['val_E_loss'])
plt.subplot(1,3,3)
plt.plot(history.history['lnA_loss'])
plt.plot(history.history['val_lnA_loss'])

score = model.evaluate(x_test_std, [y_test_label, y_test[:,1], y_test[:,2]])
print("The accuracy is : {}".format(score[4]))

#------------------------------------------------------------------------------
# t-sne visualization before classification
#------------------------------------------------------------------------------
from sklearn.manifold import TSNE
vis = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer('dense_1').output])

def view_embeded(data, label):
     plt.figure(figsize=(10,10))
     x_embedded_2d = TSNE(n_components=2, random_state=123).fit_transform(data)
     plt.scatter(x_embedded_2d[:, 0], x_embedded_2d[:, 1], 25, c=label, cmap = 'rainbow')
     plt.colorbar()
     
#view_embeded(vis([x_train_std,0])[0], np.argmax(y_train_label, axis = 1))

#Fine_tune = False
#if Fine_tune:
#    # Fine tune the regression part
#    for i in range(len(model.layers)-1):
#        model.layers[i].trainable = False
#        
#    
#    model.compile(loss ={'which_model': 'categorical_crossentropy', 
#                         'ElnA': 'mean_squared_logarithmic_error'},
#                  loss_weights={'which_model': 0.0, 'ElnA': 1.0},
#                  optimizer = optimizers.SGD(lr=0.005, decay=1e-4, momentum=0.9, nesterov=True),
#                metrics = {'which_model': 'accuracy'}
#                )
#    
#    history2= model.fit(x=x_train_std, y= [y_train_label, y_train[:,1:]],
#                        batch_size = batchsize,
#                        epochs = 20, verbose = 0,
#                        validation_data = (x_test_std, [y_test_label, y_test[:,1:]])
#                        )
#    
#    
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.plot(history2.history['which_model_loss'])
#    plt.plot(history2.history['val_which_model_loss'])
#    plt.subplot(1,2,2)
#    plt.plot(history2.history['ElnA_loss'])
#    plt.plot(history2.history['val_ElnA_loss'])
#    
#    score = model.evaluate(x_test_std, [y_test_label, y_test[:,1:]])
#    
#    print("The accuracy is : {}".format(score[3]))
# -*- coding: utf-8 -*-
from keras.layers import Input, Conv2D,Conv1D, Lambda, merge, Dense, Flatten,MaxPooling2D,MaxPooling1D,Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import time
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_siamese_net(input_shape = (2048,1)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    convnet = Sequential()
    # WDCNN
    convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same', kernel_constraint=regk))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
#     print('WDCNN convnet summary:')
#     convnet.summary()
    #call the convnet Sequential model on each of the input tensors so params will be shared
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    
    #============Euclidean============
    #L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    
    #==============MeanRelativeError======
    #L1_layer = Lambda(lambda tensors:K.abs(100*K.ones_like(tensors[0]) * (tensors[0] - tensors[1]) / tensors[0]))
    
    #===========Pearson Correlation Coefficient============
    L1_layer = Lambda(pearson_r, output_shape=eucl_dist_output_shape)([encoded_l,encoded_r])
    
    #============Cosine similarity===========
    #L1_layer = Lambda(_cosine, output_shape=eucl_dist_output_shape)([encoded_l,encoded_r])
    
    #call this layer on list of two input tensors.
   # L1_distance = L1_layer([encoded_l, encoded_r])
   # D1_layer = Dropout(0.5)(L1_distance)
   # prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=L1_layer)
    # optimizer = Adam(0.00006)
    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    #siamese_net.compile(loss=euclideann,optimizer=optimizer)
    #siamese_net.compile(loss=euclideann, optimizer=optimizer)
    
#     print('\nsiamese_net summary:')
#     siamese_net.summary()
#     print(siamese_net.count_params())
    return siamese_net

def _cosine(x):
    dot1 = K.batch_dot(x[0], x[1], axes=1)
    dot2 = K.batch_dot(x[0], x[0], axes=1)
    dot3 = K.batch_dot(x[1], x[1], axes=1)
    max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
    print("================dot1 / max_=====================")
    print(dot1)
    print(max_)
    print(dot1 / max_)
    return dot1 / max_
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print("================shape1[0], 1=====================")
    print(shape1[0], 1)
    return (shape1[0], 1)

def pearson_r(P):
    x = P[0]
    y = P[1]
    mx = K.mean(x, axis=1, keepdims=True)
    print(mx)
    my = K.mean(y, axis=1, keepdims=True)
    print(my)
    xm, ym = x - mx, y - my
    print(xm,ym)
    r_num = K.sum(xm * ym, axis=1, keepdims=True)
    print(r_num)
    x_square_sum = K.sum(xm * xm, axis=1, keepdims=True)
    y_square_sum = K.sum(ym * ym, axis=1, keepdims=True)
    print(x_square_sum, y_square_sum)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    print(r_den)
    r = r_num / r_den
    print(r)
    return r


#2019.11.18 by yfshi
#判别器，判别输入数据是来自源域还是目标域
def discriminator():
    model = Sequential()
    model.add(layers.Dense(128))        
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1))
    feature = layers.Input(shape=(self.feature_dim,))
    validity = model(feature)
    model.summary()
    return Model(feature, validity)
#2019.11.18 by yfshi
#分类器，在轴承数据对下标签类别相同时，或者在不相同时，对故障进行分类
def build_classifier(self):
    model = Sequential()
    model.add(layers.Dense(self.nb_classes, activation='softmax'))
    feature = layers.Input(shape=(self.feature_dim,))
    cls = model(feature)
    model.summary()
    return Model(feature, cls)

# def contrastive_loss(y_true, y_pred):
#     margin_right = 1
#     margin_left = 0.2
#     return K.mean(y_true * K.square(K.maximum(y_pred,margin_left)) + (1 - y_true) * K.square(K.maximum(margin_right - y_pred, 0)))

def binary(y_true, y_pred):
    margin = 1
#     square_pred = y_pred
    margin_square = K.minimum(y_pred, 0.7)
    return K.mean(K.binary_crossentropy(margin_square, y_true), axis=-1)
    #return K.mean(-(y_true * K.log(exp(y_pred)) + (1 - y_true) * K.log(1-exp(margin_square))))

def regk(weight_matrix):
    return 0.5 * weight_matrix


def load_wdcnn_net(input_shape = (2048,1),nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()
    
    # WDCNN
    convnet.add(Conv1D(filters=16, kernel_size=64, strides=16, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))


#     print('convnet summary:')
    # convnet.summary()


    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(nclasses,activation='softmax')(Dropout(0.5)(encoded_cnn ))
    wdcnn_net = Model(inputs=left_input,outputs=prediction_cnn)


    # optimizer = Adam(0.00006)
    optimizer = Adam()
    wdcnn_net.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    # print('\nsiamese_net summary:')
    # cnn_net.summary()
    print(wdcnn_net.count_params())
    return wdcnn_net

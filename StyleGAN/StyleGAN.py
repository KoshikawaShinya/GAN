import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape, Upsampling2D
from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from AdaIN import AdaInstanceNormalization


def g_block(input, style, noise, filter, u=True):

    b = Dense(filter)(style)
    b = Reshape([1, 1, filter])(b)
    g = Dense(filter)(style)
    g = Reshape([1, 1, filter])(g)

    n = Conv2D(filters=filter, kernel_size=3, padding='same', kernel_initializer='he_normal')(noise)

    if u:
        out = Upsampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = filter, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = Activation('linear')(input)

    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    b = Dense(filter)(style)
    b = Reshape([1, 1, filter])(b)
    g = Dense(filter)(style)
    g = Reshape([1, 1, filter])(g)

    n = Conv2D(filters = filter, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)

    out = Conv2D(filters = filter, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = AdaInstanceNormalization()([out, b, g])
    out = add([out, n])
    out = LeakyReLU(0.01)(out)

    return out


def build_generator(mapping_network, synthesisnetwork):
    
    return generator

def build_mapping_network(z_dim):

    model = Sequential()

    model.add(Dense(512, input_shape=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    return model

def AdaIN(x):

    # x[0]の正規化
    mean = K.mean(x[0], axis=[1, 2], keepdims=True)
    std = K.std(x[0], axis=[1, 2], keepdims=True) + 1e-7

    y = (x[0] - mean) / std

    # gammaとbetaのreshape
    pool_shape = [-1, 1, 1, y.shape[-1]]
    g = tf.reshape(x[1], pool_shape) + 1.0
    b = tf.reshape(x[2], pool_shape)

    # x[1] (gamma) を掛けて、x[2] (beta) を足す
    return y * g + b
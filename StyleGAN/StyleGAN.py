import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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
    # y : [0, 1]
    y = (x[0] - mean) / std

    # gammaとbetaのreshape
    pool_shape = [-1, 1, 1, y.shape[-1]]
    g = tf.reshape(x[1], pool_shape) + 1.0
    b = tf.reshape(x[2], pool_shape)

    # x[1] (gamma) を掛けて、x[2] (beta) を足す
    return y * g + b
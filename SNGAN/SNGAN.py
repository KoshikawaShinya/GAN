import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import keras as K
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from Spectral_Normalization import SpectralNormalization, SNConv2D


def build_generator(z_dim):

    model = Sequential()

    # 8x8x256
    model.add(Dense(8 * 8 * 256, input_dim=z_dim))
    model.add(Reshape((8, 8, 256)))

    # 8x8x256 => 8x8x128
    model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    # 8x8x128 => 16x16x64
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    # 16x16x64 => 32x32x32
    model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    # 32x32x32 => 64x64x16
    model.add(Conv2DTranspose(16, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.02))
    # 64x64x16 => 64x64x3
    model.add(Conv2DTranspose(3, kernel_size=3, strides=1, padding='same'))
    # tanh関数を適用して出力
    model.add(Activation('tanh'))

    return model


def build_discriminator(img_shape):

    model = Sequential()

    # 64x64x3 => 32x32x32
    model.add(SNConv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    # 32x32x32 => 16x16x64
    model.add(SNConv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    # 16x16x64 => 8x8x128
    model.add(SNConv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    # 8x8x128 => 4x4x256
    model.add(SNConv2D(256, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    return model


def train(iterations, batch_size, sample_interval):

    losses = []
    accuracies = []
    iteration_checkpoints = []

    dataset = np.load('datasets/irasutoya_64.npz')

    x_train = np.array(dataset['X_train'])

     # 本物の画像のラベルは全て1にする
    real = np.ones((batch_size, 1))

    # 偽物の画像のラベルは全て0にする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #-------------
        # 識別器の学習
        #-------------

        for i in range(5):

            # 本物の画像集合からランダムにバッチを生成する
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # 偽物の画像からなるバッチを生成する
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict(z)

            # 識別器の学習
            d_loss_real = discriminator.train_on_batch(imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real , d_loss_fake)

        #-------------
        # 生成器の学習
        #-------------

        # ノイズベクトルを生成
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        # 生成器の学習
        g_loss = gan.train_on_batch(z, real)

        print('\rNo, %d' %(iteration+1), end='')

        if (iteration + 1) % sample_interval == 0:

            # あとで可視化するために損失と精度を保存しておく
            losses.append([d_loss, g_loss])
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 学習結果の出力
            print('[D loss: %f, acc.: %.2f%%] [G loss: %f]' %(d_loss, 100.0*accuracy, g_loss))

            sample_images(generator, iteration)
        
        if (iteration + 1) % (sample_interval+1) == 0:
            sample_images()


def sample_images(generator, iteration, image_grid_rows=4, image_grid_columns=4):

    # ノイズベクトルを生成する
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ノイズベクトルから画像を生成する
    gen_imgs = generator.predict(z)

    # 出力の画素値を[0, 1]の範囲にスケーリングする
    gen_imgs = 0.5 * gen_imgs + 0.5


    # 画像からなるグリッドを生成する
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    count = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を出力
            axs[i, j].imshow(gen_imgs[count, :, :, 0])
            axs[i, j].axis('off')
            count += 1
    
    fig.savefig("generate_imgs/%d.png" % iteration+1)
    plt.close()
    

img_rows = 64
img_cols = 64
channels = 3

iterations = 10000
batch_size = 1
sample_interval = 100

img_shape = (img_rows, img_cols, channels)
z_dim = 100

# 識別器
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 生成器
generator = build_generator(z_dim)

discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

train(iterations, batch_size, sample_interval)
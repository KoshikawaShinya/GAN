import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from dataloader import DataLoader
from snconv2d import SNConv2D



def build_generator(z_dim):

    model = Sequential()

    # 8x8x128
    model.add(Dense(8 * 8 * 128, input_dim=z_dim))
    model.add(Reshape((8, 8, 128)))

    # 8x8x128 => 16x16x64
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
    
    # 16x16x64 => 32x32x32
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())

    # 32x32x32 => 64x64x16
    model.add(Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    model.add(BatchNormalization())
 


    #64x64x16 => 128x128x3
    model.add(Conv2DTranspose(3, kernel_size=3, strides=2, padding='same'))

    # tanh関数を適用して出力
    model.add(Activation('tanh'))

    model.summary()

    return model


def build_discriminator(img_shape):

    model = Sequential()

    # 128x128x3 => 64x64x16
    model.add(SNConv2D(16, kernel_size=4, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Dropout(rate=0.2))

    # 64x64x32 => 32x32x32
    model.add(SNConv2D(32, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Dropout(rate=0.2))

    # 32x32x64 => 16x16x32
    model.add(SNConv2D(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))
    #model.add(Dropout(rate=0.2))

    model.add(SNConv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.02))


    model.add(Flatten())
    #model.add(Dense(256))
    #model.add(LeakyReLU(alpha=0.02))
    #model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model


def build_gan(generator, discriminator):

    model = Sequential()

    model.add(generator)
    model.add(discriminator)

    return model


def train(data_loader, epochs, batch_size, sample_interval):

    accuracy = 0.0
    losses = []
    accuracies = []
    epochs_checkpoints = []


     # 本物の画像のラベルは全て1にする
    real = np.ones((batch_size, 1))

    # 偽物の画像のラベルは全て0にする
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        sample = 0
        for batch_i, real_imgs in enumerate(data_loader.load_batch(batch_size)):

            #-------------
            # 識別器の学習
            #-------------
            # 偽物の画像からなるバッチを生成する
            z = np.random.normal(0, 1, (batch_size, z_dim))
            gen_imgs = generator.predict(z)

            # 識別器の学習
            d_loss_real = discriminator.train_on_batch(real_imgs, real)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss, accuracy = 0.5 * np.add(d_loss_real , d_loss_fake)

            #-------------
            # 生成器の学習
            #-------------
            if batch_i % 5 == 0:
                # ノイズベクトルを生成
                z = np.random.normal(0, 1, (batch_size, z_dim))    
                # 生成器の学習
                g_loss = gan.train_on_batch(z, real)

            print('\rNo, %d_%d' %(epoch+1, batch_i), end='')

            if (epoch + 1) % sample_interval == 0 and sample == 0:

                sample += 1
                # あとで可視化するために損失と精度を保存しておく
                losses.append([d_loss, g_loss])
                accuracies.append(100.0 * accuracy)
                epochs_checkpoints.append(epoch + 1)

                # 学習結果の出力
                print('[D loss: %f, acc.: %.2f%%] [G loss: %f]' %(d_loss, 100.0*accuracy, g_loss))

                sample_images(generator, epoch)

                generator.save("saved_model/catdog/1/%d_catdog.h5" % (epoch+1))


def sample_images(generator, epoch, image_grid_rows=4, image_grid_columns=4):

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
            axs[i, j].imshow(gen_imgs[count, :, :, :])
            axs[i, j].axis('off')
            count += 1
    
    fig.savefig("generate_imgs/catdog/1/%d.png" % (epoch+1))
    plt.close()


# データローダの設定
dataset_name = 'resized_128'
# 前処理済みのデータをインポートするためにDataLoaderオブジェクトを用いる
data_loader = DataLoader(dataset_name)

img_rows = 128
img_cols = 128
channels = 3

epochs = 1000000
batch_size = 64
sample_interval = 10

img_shape = (img_rows, img_cols, channels)
z_dim = 100

# 識別器
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.9), metrics=['accuracy'])

# 生成器
generator = build_generator(z_dim)

discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.9), loss='binary_crossentropy')

train(data_loader, epochs, batch_size, sample_interval)
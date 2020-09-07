# Denosing AutoEncoder (DAE)
# MAXPOOLING2D & Upsampling 사용
# 큰 차이점 : 채널 수가 줄어든다.
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 , 28, 1) / 255.
x_test = x_test.reshape(-1, 28 , 28, 1) / 255.

def make_making_noise_data(data_x, percent= 0.1):
    size = data_x.shape
    making = np.random.binomial(n=1, p=percent, size=size)
    return data_x * making
x_train_maked = make_making_noise_data(x_train)
x_test_maked = make_making_noise_data(x_test)

def make_gaussian_noise_data(data_x, scale=0.8):
    gaussian_data_x = data_x + np.random.normal(loc=0, scale=scale, size=data_x.shape)
    gaussian_data_x = np.clip(gaussian_data_x, 0, 1)
    return gaussian_data_x

x_train_gauss = make_gaussian_noise_data(x_train)
x_test_gauss = make_gaussian_noise_data(x_test)

from tensorflow.keras.preprocessing.image import array_to_img
plt.imshow(array_to_img(x_train[0]))
plt.figure()
plt.imshow(array_to_img(x_train_gauss[0]))
plt.figure()
plt.imshow(array_to_img(x_train_maked[0]))
plt.show()

autoencoder = Sequential()
#encoder
autoencoder.add(Conv2D(16, (3,3), 1, activation='relu', padding='same', input_shape=(28, 28, 1)))
autoencoder.add(MaxPooling2D((2,2), padding='same'))
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2,2),padding='same'))
#decoder
autoencoder.add(Conv2D(8, (3,3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(16, (3,3), 1, activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Conv2D(1, (3,3), 1, activation='sigmoid', padding='same'))
autoencoder.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(x_train_gauss, x_train, epochs=20, batch_size=20, shuffle=True)
gauss_preds = autoencoder.predict(x_test_gauss)

from tensorflow.keras.preprocessing.image import array_to_img
plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(1, 3, 1)
    plt.imshow(array_to_img(x_test[i]))
    # plt.figure()
    plt.subplot(1, 3, 2)
    plt.imshow(array_to_img(x_test_gauss[i]))
    # plt.figure()
    plt.subplot(1, 3, 3)
    plt.imshow(array_to_img(gauss_preds[i]))
    plt.show()
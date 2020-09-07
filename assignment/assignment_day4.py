# ===============================================================
# 1. AutoEncoder 모델을 케라스를 이용하여 Fully Connected layer만으로 구성한다.
# 2. class AE를 구성한다.
# 3. 클래스 내부에 멤버함수 Encoder와 Decoder를 구성한다.
# 4. show_ae()메서드를 구성하여 입력 이미지와 디코더 이미지를 출력할 수 있게 한다.
# ===============================================================
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import random

# 데이터 셋 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# x_train = x_train.reshape(-1, 28 * 28)
# x_test = x_test.reshape(-1, 28 * 28)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
X_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

class AE(models.Model):
    def __init__(self, input_size, z_dim):
        x = layers.Input(shape=(input_size, ))
        layer_1 = layers.Dense(64, activation='relu')(x)
        layer_2 = layers.Dense(z_dim, activation='relu')(layer_1)
        layer_3 = layers.Dense(64, activation='relu')(layer_2)
        layer_4 = layers.Dense(input_size, activation='sigmoid')(layer_3)

        super().__init__(x, layer_4)
        self.x = x
        self.layer_2 = layer_2
        self.z_dim = z_dim
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def Encoder(self):
        return models.Model(self.x, self.layer_2)
    def Decoder(self):
        z_1 = layers.Input(shape=(self.z_dim,))
        z_2= layers.Dense(64, activation='relu')(z_1)
        z_3= layers.Dense(784, activation='sigmoid')(z_2)
        # z__ = self.layers[-1](z_)
        return models.Model(z_1, z_3)

def show_ae(model):
    rand_index = random.randint(0, x_test.shape[0])
    encoder = model.Encoder()
    decoder = model.Decoder()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[rand_index].reshape(28, 28), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    encoder_result = encoder.predict(np.expand_dims(x_test[rand_index], axis=0))
    encoder_ = encoder_result.copy()
    plt.imshow(encoder_.reshape(6, 6), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    decoder_result = decoder.predict(encoder_result)
    decoder_ = decoder_result.copy()
    plt.imshow(decoder_.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

def show_ae_(autoencoder):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def main():
    input_size = 784
    z_dim = 36
    # 모델 구현 (클래스)
    model = AE(input_size, z_dim)
    model.summary()
    # 학습
    model.fit(x_train, x_train, epochs=40, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=2)
    # 결과 출력 (인코더, 디코더)
    show_ae_(model)

if __name__ == '__main__':
    main()



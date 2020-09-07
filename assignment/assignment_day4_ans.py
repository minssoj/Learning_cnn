from tensorflow.keras import layers, models  # (Input, Dense), (Model)
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

class AE(models.Model):
    def __init__(self, x_nodes=784, h_dim=36):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape)
        h = layers.Dense(h_dim, activation='relu')(x)
        y = layers.Dense(x_nodes, activation='sigmoid')(h)

        super().__init__(x, y)

        self.x = x
        self.h = h
        self.h_dim = h_dim

        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def Encoder(self):
        return models.Model(self.x, self.h)

    def Decoder(self):
        h_shape = (self.h_dim,)
        h = layers.Input(shape=h_shape)
        y_layer = self.layers[-1]
        y = y_layer(h)
        return models.Model(h, y)


(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


def show_ae(autoencoder):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
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
    x_nodes = 784
    h_dim = 36

    autoencoder = AE(x_nodes, h_dim)

    autoencoder.fit(X_train, X_train,
                              epochs=30,
                              batch_size=256,
                              shuffle=True,
                              validation_data=(X_test, X_test))

    show_ae(autoencoder)
    plt.show()

main()



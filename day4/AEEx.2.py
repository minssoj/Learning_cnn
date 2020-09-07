# Convolutional AutoEncoder (CAE)
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Conv2DTranspose, Reshape
import numpy as np
import matplotlib.pyplot as plt
import random

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 , 28, 1) / 255.
x_test = x_test.reshape(-1, 28 , 28, 1) / 255.

# relu 대신 elu, leaky-relu (기울기 0.1부터 0.3정도까지) 등을 사용하는 것도 좋을 수 있음 (Auto Encoder의 경우)
model = Sequential([
    Conv2D(32, kernel_size=2, strides=(2,2), activation='elu', input_shape=(28, 28, 1)),
    Conv2D(64, kernel_size=2, strides=(2,2), activation='elu'),
    Flatten(),
    Dense(64, activation='elu'),
    Dense(7 * 7 * 64, activation='elu'),
    Reshape(target_shape=(7,7,64)),
    Conv2DTranspose(32, kernel_size=2, strides=(2,2), padding='same', activation='elu'),
    Conv2DTranspose(1, kernel_size=2, strides=(2,2), padding='same', activation='sigmoid')
])
model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, x_train, epochs=20, batch_size=200)

plt.figure(figsize=(4,8))
for i in range(4):
    plt.subplot(4,2,(i * 2 + 1))
    rand_index = random.randint(0, x_test.shape[0])
    plt.imshow(x_test[rand_index].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.subplot(4,2,(i * 2 + 2))
    img = model.predict(np.expand_dims(x_test[rand_index], axis=0))
    plt.imshow(img.reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()
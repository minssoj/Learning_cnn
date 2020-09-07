from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.
x_test = x_test.reshape(-1, 28 * 28) / 255.

# 모델 구현 및 학습
model = Sequential([
    Dense(784, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(784, activation='sigmoid')
])
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, x_train, epochs=20, batch_size=200)

import random
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
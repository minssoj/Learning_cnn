from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

tf.random.set_seed(3)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
result = model.fit(x_train, y_train, validation_split=0.3, epochs=20, batch_size=200)
print('accuracy:', model.evaluate(x_test, y_test)[1])

import numpy as np
y_vloss = result.history['val_loss']
y_acc = result.history['accuracy']
x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c='red', ms=2, label='validation_loss')
plt.plot(x_len, y_acc, 'o', c='blue', ms=2, label='accuracy')
plt.legend(loc='best')
plt.show()
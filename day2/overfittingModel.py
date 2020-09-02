from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

tf.random.set_seed(3)

df = pd.read_csv('dataset/wine.csv', header=None)
df = df.sample(frac=0.15)

data = df.values
x_data = data[:, 0:12]
y_data = data[:, 12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
estopcallback = EarlyStopping(monitor='val_loss', patience=100)

model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir((model_dir))
# weight만 저장(모델은 저장하지 않음)
savecallback = ModelCheckpoint(filepath='./model/{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only=True)

result = model.fit(x_data, y_data, validation_split=0.33, epochs=3500, batch_size=500, callbacks=[estopcallback, savecallback])
y_vloss = result.history['val_loss']
y_acc = result.history['accuracy']

x_len = np.arange(len(y_acc))
plt.plot(x_len, y_vloss, 'o', c='red', ms=2, label='validation_loss')
plt.plot(x_len, y_acc, 'o', c='blue', ms=2, label='accuracy')
plt.legend(loc='best')
plt.show()
'''
model.load_weights('./model/0.1335.hdf5')
print(model.evaluate(x_data, y_data))
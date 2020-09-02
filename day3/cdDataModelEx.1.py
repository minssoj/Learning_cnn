from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

img_size = 150

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

base_dir = '../datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
    #binary => 0, 1 1차원배열
    #sparse => 정수 레이블 1차원 배열
    #categorical => one hot encoding 2차원 배열
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

result = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

import matplotlib.pyplot as plt
x_len = range(len(result.history['accuracy']))
plt.plot(x_len, result.history['accuracy'], 'bo', label='train_accuracy')
plt.plot(x_len, result.history['val_accuracy'], 'ro', label='validation_accuracy')
plt.legend()
plt.title('accuracy')
plt.figure()
plt.plot(x_len, result.history['loss'], 'bo', label='train_loss')
plt.plot(x_len, result.history['val_loss'], 'ro', label='validation_loss')
plt.legend()
plt.title('loss')
plt.show()



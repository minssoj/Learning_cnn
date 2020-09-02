from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = '../datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# VGG 제외하고 Dense Layer만 학습하고 싶으면... 아래의 한줄
conv_base.trainable = False

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validata_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validata_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-4),
              metrics=['accuracy'])
result = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

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
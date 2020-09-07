# ===============================================================
# 1. 절과 신사 이미지로 구성된 데이터를 이용한 모델을 구성한다.
# 2. 모델은 VGG16을 전이 받아 구성한다. 이때 VGG16의 FC layer는 사용하지 않고 직접 생성하여 모델을 구성한다.
# 3. 모델 학습 시 VGG16 부분은 학습에서 제외하고 진행한다.
# 4. 사용할 이미지는 ImageDataGenerator를 이용하여 데이터를 증식한다.
# ===============================================================
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

base_dir = '../datasets/img/shrine_temple'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
label = ['shrine', 'temple']

# 데이터 증식
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
    target_size=(224,224),
    batch_size=20,
    class_mode='binary'
)
validata_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validata_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=20,
    class_mode='binary'
)

# 모델 구현
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model = models.Sequential()
model.add(conv_base)
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = True
l_t = ['block5_conv3']
for layer in conv_base.layers:
    set_trainable = False
    if layer.name in l_t:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-4),metrics=['accuracy'])
conv_base.summary()
model.summary()

# 학습 진행
result = model.fit_generator(
    train_generator,
    steps_per_epoch=30,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=10)

x_len = range(len(result.history['accuracy']))
plt.plot(x_len, result.history['accuracy'], 'b', label='train_accuracy')
plt.plot(x_len, result.history['val_accuracy'], 'r', label='validation_accuracy')
plt.legend()
plt.title('accuracy')
plt.figure()
plt.plot(x_len, result.history['loss'], 'b', label='train_loss')
plt.plot(x_len, result.history['val_loss'], 'r', label='validation_loss')
plt.legend()
plt.title('loss')
plt.show()

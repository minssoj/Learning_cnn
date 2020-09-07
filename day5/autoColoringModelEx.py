from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import  load_img, img_to_array, array_to_img
import cv2
import os
import math
import numpy as np
import glob
import matplotlib.pyplot as plt
data_path = './img/colorize'
data_lists = glob.glob(os.path.join(data_path, '*.jpg'))
print(data_lists)
val_n_sample = math.floor(len(data_lists)*0.1)
test_n_sample = math.floor(len(data_lists)*0.1)
train_n_sameple = len(data_lists) - val_n_sample - test_n_sample

val_lists = data_lists[:val_n_sample]
test_lists = data_lists[val_n_sample : val_n_sample + test_n_sample]
train_lists = data_lists[val_n_sample + test_n_sample : val_n_sample + test_n_sample + train_n_sameple]
img_size = 224

def rgb2lab(rgb):
    assert rgb.dtype == 'uint8'
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
def lab2rgb(lab):
    assert lab.dtype == 'uint8'
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
def get_lab_from_data_list(data_list):
    x_lab = []
    for f in data_list:
        rgb = img_to_array(load_img(f, target_size=(img_size, img_size))).astype(np.uint8)
        lab = rgb2lab(rgb)
        x_lab.append(lab)
    return np.stack(x_lab)

autoencoder = Sequential()
autoencoder.add(Conv2D(32, (3,3), (1,1), activation='relu', padding='same', input_shape=(224,224,1)))
autoencoder.add(Conv2D(64, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2D(128, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2D(256, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(128, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(64, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2DTranspose(32, (3,3), (2,2), activation='relu', padding='same'))
autoencoder.add(Conv2D(2, (1,1), (1,1), activation='relu', padding='same'))

def generator_with_preprocessing(data_list, batch_size, shuffle=False):
    while True:
        if shuffle:
            np.random.shuffle(data_list)
        for i in range(0, len(data_list), batch_size):
            batch_list = data_list[i: i+batch_size]
            batch_lab = get_lab_from_data_list(batch_list)
            batch_l = batch_lab[:, :, :, 0:1] #L만 추출
            batch_ab = batch_lab[:, :, :, 1:] #AB 추출
            yield (batch_l, batch_ab)
batch_size = 20
train_gen = generator_with_preprocessing(train_lists, batch_size, shuffle=True)
val_gen = generator_with_preprocessing(val_lists, batch_size, shuffle=True)
train_steps = math.ceil(len(train_lists)/ batch_size)
val_steps = math.ceil(len(val_lists)/ batch_size)
test_steps = math.ceil(len(test_lists)/ batch_size)

autoencoder.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_steps,
    epochs=3,
    validation_data=val_gen,
    validation_steps=val_steps
)
x_test = []
y_test = []
for i, (l, ab) in enumerate(generator_with_preprocessing(test_lists, batch_size)):
    x_test.append(l)
    y_test.append(ab)
    if i == (test_steps - 1):
        break

x_test = np.vstack(x_test).astype(np.float32)
y_test = np.vstack(y_test)
preds = autoencoder.predict(x_test)
test_preds_lab = np.concatenate((x_test, preds), 3).astype(np.uint8)

test_preds_rgb = []
for i in range(test_preds_lab.shape[0]):
    preds_rgb = lab2rgb(test_preds_lab[i, :, :, :])
    test_preds_rgb.append(preds_rgb)
test_preds_rgb = np.stack(test_preds_rgb)
from PIL import ImageOps
for i in range(test_preds_rgb.shape[0]):
    gray_image = ImageOps.grayscale(array_to_img(test_preds_rgb[i]))
    plt.imshow(gray_image)
    plt.figure()
    plt.imshow(array_to_img(test_preds_rgb[i]))
    plt.show()
    if i == 10:
        break


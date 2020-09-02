# ===============================================================
# 1. CIFAR-10 데이터를 이용하여 10가지 물체를 구별할 수 있는 dnn 모델과 cnn 모델을 케라스를 이용하여 생성한다.
#    from tensorflow.keras.datasets import cifar10
#    (X_train, y_train), (X_test, y_test) = cifar10.load_data() #size: (32 * 32)
#    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# 2. 데이터를 통해 학습 후 각 각 두 모델을 저장한다.
# 3. 저장된 모델을 읽어 들여 제공한 이미지 파일을 판별하는 프로그램을 구성한다.
# ===============================================================
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, MaxPooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# training(DNN) : 0, training(CNN) :1,  prediction : 2
option = 1

# 데이터셋 준비

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #size: (32 * 32)
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(y_train[0])

# 모델 구현
if option == 0:
    x_train = x_train.reshape(x_train.shape[0], (32*32*3)).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], (32*32*3)).astype('float32') / 255
    model = models.Sequential()
    model.add(Dense(512, input_dim=(32 * 32 * 3), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    file_name = './day3_dnn.h5'

elif option == 1:
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    x_train = x_train / 255.
    x_test = x_test / 255.
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(32, 32, 3))
    conv_base.trainable = True
    for layer in conv_base.layers:
        set_trainable = False
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-4),
                  metrics=['accuracy'])
    '''
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    '''
    file_name = './day3_cnn.h5'
if option == 0 or option == 1:
    result = model.fit(x_train, y_train, epochs=50, validation_split=0.25, batch_size=32)
    model.save(file_name)

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

if option == 2:
    models_dnn = load_model('./day3_dnn.h5')
    models_cnn = load_model('./day3_cnn.h5')
    img_path = './datasets/test-car.jpg'

    img = image.load_img(img_path, target_size=(32,32,3))
    img_tensor = image.img_to_array(img)
    # print(img_tensor.shape)

    img_tensor_dnn = img_tensor.reshape(1, (32 * 32 * 3)).astype('float32') / 255
    output_dnn = models_dnn.predict(img_tensor_dnn)
    print("DNN 결과:", labels[np.argmax(output_dnn)])


    img_tensor_cnn = np.expand_dims(img_tensor, axis=0)
    img_tensor_cnn /= 255.

    output_cnn = models_cnn.predict(img_tensor_cnn)
    print("CNN 결과:", labels[np.argmax(output_cnn)])
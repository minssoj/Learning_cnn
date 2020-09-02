# ===============================================================
# 1. mnist 이미지 데이터를 이용하여 CNN 모델을 구성한다.
# 2. 이 때 CNN 모델을 클래스(Sequential 상속)로 구성한다.
# 3. 학습이 끝난 후 모델을 평가하여 loss와 accuracy를 출력한다.
# ===============================================================
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

# 데이터 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train / 255.
x_test = x_test / 255.

# 모델 구현
class mnistCNN(Sequential):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.add(Conv2D(input_shape=(input_size, input_size, 1),
                        kernel_size=(3,3), filters=64, padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=2))
        self.add(Conv2D(kernel_size=(3,3), filters=32, padding='same', activation='relu'))
        self.add(MaxPool2D(pool_size=2))
        self.add(Dropout(0.25))
        self.add(Flatten())
        self.add(Dense(units=128, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(output_size, activation='softmax'))
        self.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

model = mnistCNN(28, 10)
# 학습
result = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=100)

# 결과 출력
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(result.history['loss'], 'b-', label='loss')
plt.plot(result.history['val_loss'], 'r-', label='val_loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(result.history['accuracy'], 'b-', label='accuracy')
plt.plot(result.history['val_accuracy'], 'r-', label='val_accuracy')
plt.xlabel('epoch')
plt.ylim(0.7, 1)
plt.legend()
plt.show()
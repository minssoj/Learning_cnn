from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import tensorflow as tf

tf.random.set_seed(3)

df = pd.read_csv('dataset/sonar.csv', header=None)
data = df.values
x_data = data[:, 0:60].astype(float)
y_data = data[:, 60]

le = LabelEncoder()
le.fit(y_data)
y_data = le.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=3)

# 모델 구현
model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=5)

model.save('./model/sonarModel.h5')
del model

from tensorflow.keras.models import load_model

model = load_model('./model/sonarModel.h5')
e_result = model.evaluate(x_test, y_test)
print(e_result)
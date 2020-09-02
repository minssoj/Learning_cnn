# tensorEx.5.py
# tensorflow 1점대 코드

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.constant(5)
b = tf.constant(10)

c = a*b
print(c, '\n')

sess = tf.Session()
result = sess.run(c)
print(result,'\n')
print(sess.run([a,b,c]), '\n')

input_data = [2,6,3,8,9]
x = tf.placeholder(dtype=tf.float32)
y = x*2

sess = tf.Session()
# y를 실행하는 순간 placeholder에 데이터를 넣어준다.
sess.run(y,feed_dict={x:input_data})

import numpy as np
np.random.seed(12345)
x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

x = tf.placeholder(dtype=tf.float32, shape = [5,10])
w = tf.placeholder(dtype=tf.float32, shape = [10,1])
# 주의 : -1.0으로 넣어야함 x가 float32형이기에
b = tf.fill([5,1], -1.0)
output = tf.matmul(x,w)+b
result3 = tf.reduce_max(output)

print(sess.run(result3, feed_dict={x: x_data, w:w_data}), '\n')




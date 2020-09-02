import tensorflow as tf

a = tf.range(6, dtype=tf.int32)
b = tf.range(6, dtype=tf.int32) * 2
print(a)
print('\n', b)
print('\n', tf.add(a, b).numpy())
print('\n', (a+b).numpy())

# tf.reduce_ ~ : 축소해서 scalar값으로
print('\n', tf.reduce_sum(a).numpy())

c = tf.constant([[2,5,4], [7,8,9]])
print('\n', c)
print('\n', tf.reduce_mean(c, axis=1).numpy())

d = tf.constant([[2,0],[0,1]], dtype=tf.float32)
e = tf.constant([[1,1],[1,1]], dtype=tf.float32)
print('\n', tf.matmul(d, e).numpy())
print('\n', tf.linalg.inv(d).numpy())
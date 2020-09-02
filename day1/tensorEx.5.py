# tensorEx.5.py
import tensorflow as tf

x = tf.Variable(tf.constant(1.0))
with tf.GradientTape() as tape:
    y = tf.multiply(5,x)

gradient = tape.gradient(y,x)
print(gradient)

x1 = tf.Variable(tf.constant(1.0))
x2 = tf.Variable(tf.constant(2.0))
with tf.GradientTape() as tape:
    y = tf.multiply(x1,x2)

gradient = tape.gradient(y,[x1,x2])
print(gradient)
print(gradient[0].numpy())
print(gradient[1].numpy())

x3 = tf.Variable(tf.constant(1.))
a = tf.constant(2.)
with tf.GradientTape() as tape:
    tape.watch(a)
    y = tf.multiply(a, x3)

gradient2 = tape.gradient(y,a)      # None이 뜨는 이유는 상수로 미분해주기 때문... -> tape.watch 넣어주자
print(gradient2)
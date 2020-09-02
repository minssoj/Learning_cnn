import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)

x_data = [1,2,3]
y_data = [1,2,3]

w = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

hypothesis = x_data * w + b
cost = tf.reduce_mean(tf.square(hypothesis-y))
learning_rate = 0.1

'''
gradient = tf.reduce_mean((w * x - y) * x)
descent = w - learning_rate * gradient
update = w.assign(descent)
'''
update = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    # tf 1과 2의 차이 : 1은 변수를 초기화 해줘야한다.
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        _, _w, _b = sess.run([update, w, b], feed_dict={x:x_data, y:y_data})
        print('epoch:{} w:{} b:{}'.format(epoch+1, _w, _b))


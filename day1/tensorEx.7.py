# tensorEx.7
import tensorflow as tf

data1 = tf.data.Dataset.from_tensor_slices([8,3,0,5,2,1])
print(data1)
print()
for data in data1:
    print(data)

data2 = tf.data.Dataset.from_tensor_slices(
    (tf.random.uniform([4,10], maxval=10, dtype=tf.int32),
     tf.random.uniform([4]))
)
print('\n')
for d in data2:
    print(d)

def cfunc(stop):
    i = 0
    while i < stop:
        # yield 제네레이터 객체
        yield  i
        i += 1

data3 = tf.data.Dataset.from_generator(
    cfunc,
    args=[4],
    output_types=tf.int32
)
print()
for d in data3:
    print(d)
print()
for d in data3.repeat(2):
    print(d)
print()
for d in data3.batch(2):
    print(d)

x_data = [[10,20,30,40],
          [50,60,70,80],
          [90,100,110,120],
          [130,140,150,160],
          [170,180,190,200]]
y_data = [[1],[0],[1],[1],[0]]
def mapping_f(x, y):
    input = {'x': x}
    label = {'y':y}
    return input, label
data4 = tf.data.Dataset.from_tensor_slices((x_data, y_data))
data4 = data4.repeat(2)
data4 = data4.batch(2)
data4 = data4.map(mapping_f)
for data in data3:
    try:
        print(data)
    except tf.errors.OutOfRangeError:
        break
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

a = tf.range(1.0, 13.0, 1)
b = tf.range(10.0, 22.0, 1)

a = tf.reshape(a, [-1, 4])
b = tf.reshape(b, [-1, 4])

c = b / a
d = tf.reduce_sum(c, axis=1)
d = tf.expand_dims(d, axis=1)
e = tf.divide(c, d)

print(e)

print(tf.reduce_sum(tf.reduce_sum(e, axis=1)))



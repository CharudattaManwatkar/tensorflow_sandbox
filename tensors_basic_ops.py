import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.random.normal((3, 4))
y = tf.random.normal((4, 3), mean=4, stddev=2)

print(x)
print(y)
print(x @ y)
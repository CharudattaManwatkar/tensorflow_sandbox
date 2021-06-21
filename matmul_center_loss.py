import numpy as np
import tensorflow as tf

# y_true = tf.array([[1, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 1],
#                    [1, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 1]])

# nums = np.sum(y_true, axis=0)
# nums = np.expand_dims(nums, axis=0)
# print('nums\n', nums)

# x = np.array([[1, 2, 3, 4, 5],
#               [11, 12, 13, 14, 15],
#               [21, 22, 23, 24, 25],
#               [31, 32, 33, 34, 35],
#               [41, 42, 43, 44, 45],
#               [51, 52, 53, 54, 55]])

# centres = np.transpose(np.matmul(np.transpose(x), y_true))
# print('centres are\n', centres)

# centres = np.divide(centres, np.transpose(nums))
# print('centres after divide\n', centres)

# prep_centres = np.matmul(y_true, centres)

# print('prep centres\n', prep_centres)

y_true = tf.constant([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=tf.float32)

nums = tf.reduce_sum(y_true, axis=0)
nums = tf.expand_dims(nums, axis=0)
nums = tf.transpose(nums)
print('nums\n', nums)

x = tf.constant([[1, 2, 3, 4, 5],
                 [11, 12, 13, 14, 15],
                 [21, 22, 23, 24, 25],
                 [1, 2, 3, 4, 5],
                 [12, 13, 14, 15, 16],
                 [24, 25, 26, 27, 28]], dtype=tf.float32)

centres = tf.transpose(tf.matmul(tf.transpose(x), y_true))
print('centres are\n', centres)

centres = tf.divide(centres, nums)
print('centres after divide\n', centres)

prep_centres = tf.matmul(y_true, centres)

print('prep centres\n', prep_centres)

print('norms\n', tf.square(tf.norm(x - prep_centres, axis=1)))
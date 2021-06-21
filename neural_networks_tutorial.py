import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(148, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(28, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(10)
    ]
)

print(model.summary())
# import sys
# sys.exit()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(x_train, y_train, batch_size=32, verbose=2)
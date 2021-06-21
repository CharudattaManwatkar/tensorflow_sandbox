import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def data_generator(x_train, y_train, batch_size):
    i = 0
    while True:
        if i + batch_size < len(x_train):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            i += batch_size
        else:
            x_batch = np.concatenate((x_train[i:], x_train[:i+batch_size-len(x_train)]))
            y_batch = np.concatenate((y_train[i:], y_train[:i+batch_size-len(y_train)]))
            i += batch_size - len(x_train)

        yield x_batch, y_batch

x = np.random.random((100, 3))
y = np.random.randint(2, size=(100, 1))

my_gen = data_generator(x, y, 10)


inputs = tf.keras.layers.Input(shape=(3,))
outputs = tf.keras.layers.Dense(2)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])

# history = model.fit(x, y, epochs=10)
history = model.fit_generator(my_gen, epochs=10, steps_per_epoch=50)

plt.plot(history.history['accuracy'], label='Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
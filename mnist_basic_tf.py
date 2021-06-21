import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt


def data_generator(x_train, y_train, batch_size):
    i = 0
    while True:
        if i + batch_size < len(x_train):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            i += batch_size
        else:
            x_batch = np.concatenate((x_train[i:], x_train[:i+batch_size-len(y_train)]))
            y_batch = np.concatenate((y_train[i:], y_train[:i+batch_size-len(y_train)]))
            i += batch_size - len(x_train)

        yield x_batch, y_batch

def make_model():

    cnn = models.Sequential()

    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.BatchNormalization())

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation='relu'))
    cnn.add(layers.Dense(32, activation='relu'))
    cnn.add(layers.Dense(16, activation='relu'))
    cnn.add(layers.Dense(10))

    # input_one = tf.keras.Input((28, 28))
    # input_two = tf.keras.Input((512, 512, 3))

    # embedding_one = cnn(input_one)
    # embedding_two = cnn(input_two)

    # subtracted = layers.Subtract()([embedding_one, embedding_two])
    # dense_1 = layers.Dense(8, activation='relu')(subtracted)
    # dense_2 = layers.Dense(4, activation='relu')(dense_1)

    # prediction = layers.Dense(1, activation='sigmoid')(subtracted) #(dense_2)

    # siamese_net = models.Model(inputs=[input_one, input_two],
    #                             outputs=prediction)

    optimizer_cnn = tf.keras.optimizers.SGD()
    cnn.compile(loss="categorical_crossentropy", optimizer=optimizer_cnn,
                metrics=["accuracy"])

    # optimizer_siamese = tf.keras.optimizers.SGD()
    # siamese_net.compile(loss='categorical_crossentropy',
    #                     optimizer=optimizer_siamese, metrics=["accuracy"])

    return cnn

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(type(x_train))
print(x_train.shape)


# my_gen = data_generator(x_train, y_train, batch_size=64)

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])

# predictions = model(x_train[:1]).numpy()
# print(predictions)

# tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# print(loss_fn(y_train[:1], predictions).numpy())

# model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model = make_model()
# history = model.fit(my_gen, epochs=16, steps_per_epoch=32)
history = model.fit(x_train, y_train, epochs=16, steps_per_epoch=32)

model.evaluate(x_test,  y_test, verbose=2)

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# probability_model(x_test[:5])

# img = next(my_gen)[0][0]
# # plt.imshow(img)
# cv2.imshow('window_name', img)
# cv2.waitKey(0)

plt.plot(history.history['accuracy'], label='Train accuracy')
# plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

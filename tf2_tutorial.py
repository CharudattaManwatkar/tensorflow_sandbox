import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt



# x = tf.constant([1, 2, 3, 4, 5])
# y = tf.constant([1, 1, 1, 1, 1])
# a = tf.add(x, y)
# print(a)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
print('type:', type(train_images))
print('shape:', train_images.shape)



# class_names = ['zero', 'one', 'two', 'three', 'four', 'five',
#                'six', 'seven', 'eight', 'nine']

# plt.figure(figsize=(10,10))
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

IMG_SIZE = (28, 28, 1)
input_img = layers.Input(shape=IMG_SIZE)

model = layers.Conv2D(32, (3, 3), padding='same')(input_img)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.Conv2D(32, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.Conv2D(64, (3, 3), padding='same', strides=(2, 2))(model)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.Conv2D(64, (3, 3), padding='same')(model)
model = layers.Activation('relu')(model)
model = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")(model)

model = layers.GlobalAveragePooling2D()(model)
model = layers.Dense(32)(model)
model = layers.Activation('relu')(model)
model = layers.Dense(10)(model)

output_img = layers.Activation('softmax')(model)

model = models.Model(input_img, output_img)

model.summary()


train_images = train_images.reshape(60000, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(10000, 28, 28, 1).astype('float32') / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

adam = optimizers.Adam(lr=0.0001)
model.compile(adam, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(train_images, train_labels, batch_size=245, epochs=20,
                    validation_data=(test_images, test_labels))

plt.figure(figsize=(5, 5))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.01, 1])
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()


test_loss, test_accuracy = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy = {0:.2f}%'.format(test_accuracy*100.0))

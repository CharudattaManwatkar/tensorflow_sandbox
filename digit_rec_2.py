import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import glob

tf.compat.v1.set_random_seed(123)
np.random.seed(123)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def trim_dataset(full_data, f):
    if f >= 0  and f <= 1:
        return full_data[:int(full_data.shape[0]*f)]
    else:
        return full_data

x_train = trim_dataset(x_train, 0.9)
x_test = trim_dataset(x_test, 0.9)
y_train = trim_dataset(y_train, 0.9)
y_test = trim_dataset(y_test, 0.9)


# to switch test and train data if needed
switch = 0
if (switch):
    (x_train, y_train), (x_test, y_test) = (x_test, y_test), (x_train, y_train) 


print("x_train", x_train.shape, "y_train", y_train.shape, "x_test", x_test.shape, "y_test", y_test.shape)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def make_one_hot(normal_array):
    temp_array = np.empty((normal_array.size, normal_array.max()+1))
    temp_array[np.arange(normal_array.size), normal_array] = 1
    return temp_array

y_train = make_one_hot(y_train)
y_test = make_one_hot(y_test)

# Some visualization of the trainign data
# fig, axs = plt.subplots(3, 3)
# fig.suptitle('Some training sample')
# for i in range(3):
#     for j in range(3):
#         axs[i][j].imshow(x_train[(i+1)*(j+1)], cmap='gray')

# Flatten the images to feed them to a fully connected layer
x_train_flat = np.empty((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test_flat = np.empty((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

for i in range(x_train.shape[0]):
    x_train_flat[i] = x_train[i].flatten()

for i in range(x_test.shape[0]):
    x_test_flat[i] = x_test[i].flatten()

# initializing parameters for the model
batch = 100
learning_rate = 0.01
training_epochs = 100
epoch_checkpoints = 20
test_accuracy_list = []
train_accuracy_list = []
current_epoch = []

# creating placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# creating weights and biases
W1 = tf.Variable(tf.random.normal([784, 183]))
b1 = tf.Variable(tf.random.normal([183]))

W2 = tf.Variable(tf.random.normal([183, 43]))
b2 = tf.Variable(tf.random.normal([43]))

W3 = tf.Variable(tf.random.normal([43, 10]))
b3 = tf.Variable(tf.random.normal([10]))

# initializing the model
def feed_forward(x):
    z1 = tf.nn.softmax(tf.matmul(x, W1) + b1)
    z2 = tf.nn.softmax(tf.matmul(z1, W2) + b2)
    y = tf.nn.softmax(tf.matmul(z2, W3) + b3)
    return y

y = feed_forward(x)

# Defining Cost Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(y), reduction_indices=[1]))

# Determining the accuracy of parameters
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Implementing Gradient Descent Algorithm
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Initializing the session
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('session running')
    # Creating batches of data for epochs
    for epoch in range(training_epochs):
        batch_count = int(x_train_flat.shape[0] / batch)
        for i in range(batch_count):
            batch_x, batch_y = x_train_flat[i*batch:(i+1)*batch], y_train[i*batch:(i+1)*batch]

            # Executing the model
            sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})

        # Print Accuracy of the model
        if epoch % (training_epochs//epoch_checkpoints) == 0:
            train_accuracy = accuracy.eval(feed_dict={x: x_train_flat, y_: y_train})
            test_accuracy = accuracy.eval(feed_dict={x: x_test_flat, y_: y_test})
            print("Epoch: ", epoch, "/", training_epochs, "--> Test_Accuracy =", test_accuracy, "Train_Accuracy =", train_accuracy)
            test_accuracy_list.append(test_accuracy)
            train_accuracy_list.append(train_accuracy)
            current_epoch.append(epoch)
    print("Model Execution Complete")

    fig = plt.figure()
    plt.scatter(current_epoch, test_accuracy_list, label='Test Accuracy')
    plt.scatter(current_epoch, train_accuracy_list, marker='+', c='r', label='Training Accuracy')
    plot_text = "Training Set size = {} \nTesting set size = {}".format(str(x_train.shape[0]), str(x_test.shape[0])) 
    plt.title(plot_text)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    # plt.text(0.75*training_epochs, min(np.min(train_accuracy_list), np.min(test_accuracy_list)), s=plot_text, fontsize=9)
    plt.legend()
    plt.show()

    try:
        for picture in glob.glob("*.png"):
            im = cv2.imread(picture, 0)

            # invert images because I created created them on a white background
            im = np.max(im) - im
            
            # Normalization for the images
            L2 = np.atleast_1d(np.linalg.norm(im, ord=2, axis=1))
            L2[L2 == 0] = 1
            im = im / np.expand_dims(L2, axis=1)

            im = np.reshape(im, (1, 784))
            im = tf.convert_to_tensor(im, dtype=tf.float32)
            im_pred = feed_forward(im)
            answer = sess.run(im_pred)
            print(picture, np.argmax(answer, axis=1))
    except Exception as e:
        print(str(e))

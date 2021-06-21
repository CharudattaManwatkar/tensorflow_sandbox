
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

# your credentials
run = neptune.init(project='common/tf-keras-integration',
                   api_token='ANONYMOUS')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

model_input = tf.keras.Input((28, 28))
flat = tf.keras.layers.Flatten()(model_input)
dense_1 = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)(flat)
dense_2 = tf.keras.layers.Dense(10, activation='sigmoid')(dense_1)
dense_2.trainable = False

model = tf.keras.models.Model(inputs=model_input, outputs=[dense_1, dense_2])
model.build(input_shape=(28, 28))
model.summary()


def CosFaceLoss(W, b, m, s):

    def inner(y_true, x):

        # replace 0 => 1 and 1=> m in y_true
        y_true = tf.cast(y_true, dtype=tf.float32)
        M = m * y_true

        # W . x = ||W|| * ||x|| * cos(theta)
        # so (W . x) / (||W|| * ||x||) = cos(theta)

        dot_product = tf.matmul(x, W)
        cos_theta, cos_theta_norms = tf.linalg.normalize(dot_product, axis=0)

        # re-scale the cosines by a hyper-parameter s
        # and subtract appropriate margin
        y_pred = s * cos_theta - M
        y_pred = tf.matmul(x, W) + b

        # the following part is the same as softmax loss
        numerators = tf.reduce_sum(y_true * tf.exp(y_pred), axis=1)
        denominators = tf.reduce_sum(tf.exp(y_pred), axis=1)
        loss = - tf.reduce_sum(tf.math.log(numerators/denominators))

        return loss

    return inner


def ArcFaceLoss(W, m):
    def inner(y_true, x):
        # replace 0 => 1 and 1=> m in y_true
        M = (m-1) * y_true + 1

        # consider normalized weight matrix and feature vectors
        normalized_W, norms_w = tf.linalg.normalize(W, axis=0)
        normalized_x, norms_x = tf.linalg.normalize(x, axis=0)

        # W . x = ||W||*||x||*cos(theta)
        # but ||W|| = 1 and ||x|| = 1
        # so (W . x) = cos(theta)
        # cos_theta = normalized_x * normalized_W

        dot_product = tf.matmul(x, W)
        cos_theta, cos_theta_norms = tf.linalg.normalize(dot_product, axis=0)

        theta = tf.acos(cos_theta)

        # add appropriate margin to theta
        new_theta = theta + M
        new_cos_theta = tf.cos(new_theta)

        # re-scale the cosines by a hyper-parameter s
        s = 1
        y_pred = new_cos_theta

        # the following part is the same as softmax loss
        numerators = tf.reduce_sum(y_true * tf.exp(y_pred), axis=1)
        denominators = tf.reduce_sum(tf.exp(y_pred), axis=1)
        loss = - tf.reduce_sum(tf.math.log(numerators / denominators))

        return loss

    return inner


def dummy_loss(ytrue, ypred):
    return tf.constant([0])


# loss_func = CosFaceLoss(W=model.layers[-1].weights[0], m=10.0, s=10.0)
loss_func = ArcFaceLoss(W=model.layers[-1].weights[0], m=0.2)

model.compile(optimizer='adam', metrics=['accuracy'],
              loss=[loss_func, dummy_loss])

neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          callbacks=[neptune_cbk])

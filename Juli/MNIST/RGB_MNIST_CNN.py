import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 28, 28, 3])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
training_epochs = 20


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):

    w_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 32]))
    w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    w_fc = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
    w_out = tf.Variable(tf.random_normal([1024, n_classes]))

    b_conv1 = tf.Variable(tf.random_normal([32]))
    b_conv2 = tf.Variable(tf.random_normal([64]))
    b_fc = tf.Variable(tf.random_normal([1024]))
    b_out = tf.Variable(tf.random_normal([n_classes]))

    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(convolution(conv1, w_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)
    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, w_fc) + b_fc)
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, w_out) + b_out

    return output


def train_network(x):
    prediction = cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples / batch_size)):
                batch_x, batch_y = parse_batch()
                i, c = sess.run([optimizer, cost], feed_dict={x: preprocess_batch(batch_x), y: batch_y})
                epoch_loss += c

            print('Epoch', epoch, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({x: preprocess_batch(mnist.test.images), y: mnist.test.labels})
        print('Accuracy:', acc)


def parse_batch():
    parse_batch_x, parse_batch_y = mnist.train.next_batch(batch_size)
    return parse_batch_x, parse_batch_y


def preprocess_image(array):
    output = np.array([array, array, array])  # turn mnist into "RGB" file
    output = np.reshape(output, [-1, 28, 28, 3])
    return output


def preprocess_batch(input_batch):
    if input_batch.shape[0] > 1000:
        print("Starting compression of ", input_batch.shape[0], " images. This may take a while...")
    for i in range(input_batch.shape[0]):
        image = input_batch[i, :]
        image = np.reshape(image, [28, 28])
        compressed_image = preprocess_image(image)
        if i == 0:
            output_batch = compressed_image
        else:
            output_batch = np.vstack((output_batch, compressed_image))
    return output_batch


train_network(x)

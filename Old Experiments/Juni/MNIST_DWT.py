import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from skimage.transform import resize
import pywt


# dataset parameters
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
number_of_inputs = 784
conv2_shape = 7


n_classes = 10

# network placeholders
x = tf.placeholder('float', [None, number_of_inputs])
y = tf.placeholder('float')

# dropout parameters
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# training parameters
batch_size = 128
visualize_bool = 0
hm_epochs = 10


def vectorize_input(matrix):
    vector = tf.reshape(matrix, [-1])
    return vector


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([conv2_shape * conv2_shape * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, int(np.sqrt(number_of_inputs)), int(np.sqrt(number_of_inputs)), 1])
    conv1 = tf.nn.relu(convolution(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    print(conv1.shape)
    conv2 = tf.nn.relu(convolution(conv1, weights['W_conv2']) + biases['b_conv2'])
    print(conv2.shape)
    conv2 = maxpool2d(conv2)
    print(conv2.shape)
    fc = tf.reshape(conv2, [-1, conv2_shape * conv2_shape * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):

    prediction = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:  # config=tf.ConfigProto(log_device_placement=True)
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0
                for i in range(int(mnist.train.num_examples / batch_size)):
                    batch_x, batch_y = parse_batch()
                    if visualize_bool:
                        if i == 1:
                            visualize_compression(batch_x)
                    i, c = sess.run([optimizer, cost], feed_dict={x: preprocess_data(batch_x), y: batch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            acc = accuracy.eval({x: preprocess_data(mnist.test.images), y: mnist.test.labels})
            print('Accuracy:', acc)
    return acc


def parse_batch():
    parse_batch_x, parse_batch_y = mnist.train.next_batch(batch_size)
    return parse_batch_x, parse_batch_y


def compress(array):
    array = np.reshape(array, [28, 28])
    wp = pywt.WaveletPacket2D(data=array, wavelet='db1', mode='symmetric')
    output = np.hstack(([np.vstack(([wp['a'].data, wp['h'].data])), np.vstack(([wp['a'].data, wp['h'].data]))]))
    output = np.reshape(output, [1, number_of_inputs])
    return output


def downsample(array):
    array = np.reshape(array, [28, 28])
    output = resize(array, (14, 14), mode='reflect')
    output = np.reshape(output, [1, 196])
    return output


def visualize_compression(input_batch):
    for i in range(4, 7):
        image = input_batch[i, :]
        compressed_image = compress(image)
        image = np.reshape(image, [28, 28])
        compressed_image = np.reshape(compressed_image, [int(np.sqrt(number_of_inputs)), int(np.sqrt(number_of_inputs))])
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 1, 1)
        plt.imshow(image)
        fig.add_subplot(2, 1, 2)
        plt.imshow(compressed_image)
        plt.show()


def preprocess_data(input_batch):
    if input_batch.shape[0] > 1000:
        print("Starting compression of ", input_batch.shape[0], " images")
    for i in range(input_batch.shape[0]):
        image = input_batch[i, :]
        compressed_image = compress(image)
        if i == 0:
            output_batch = compressed_image
        else:
            output_batch = np.vstack((output_batch, compressed_image))
    return output_batch


acc1 = train_neural_network(x)
acc2 = train_neural_network(x)
acc3 = train_neural_network(x)
acc4 = train_neural_network(x)

acc = acc1+acc2+acc3+acc4
acc = acc/4
print(acc)

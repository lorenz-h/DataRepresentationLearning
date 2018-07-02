import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import sys
from skimage.transform import resize
import pywt
from FFST import (scalesShearsAndSpectra,
                  inverseShearletTransformSpect,
                  shearletTransformSpect)
from FFST._fft import ifftnc # centered nD inverse FFT

gpu_ident = int(sys.argv[1])
print(gpu_ident)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128
input_shape = [14, 14]

x = tf.placeholder('float', [None, input_shape[0]*input_shape[1]])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)
training_epochs = 20
n_evaluations = 3


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, input_shape[0], input_shape[1], 1])

    w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    b_conv1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    b_conv2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(convolution(conv1, w_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)
    conv2_shape = [conv2.shape[1].value, conv2.shape[2].value, conv2.shape[3].value]
    w_fc = tf.Variable(tf.random_normal([conv2_shape[0] * conv2_shape[1] * conv2_shape[2], 1024]))
    b_fc = tf.Variable(tf.random_normal([1024]))
    fc = tf.reshape(conv2, [-1, conv2_shape[0] * conv2_shape[1] * conv2_shape[2]])
    fc = tf.nn.relu(tf.matmul(fc, w_fc) + b_fc)
    fc = tf.nn.dropout(fc, keep_rate)

    w_out = tf.Variable(tf.random_normal([1024, n_classes]))
    b_out = tf.Variable(tf.random_normal([n_classes]))

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
        return acc


def test_processing():
    visualize_bool = 1
    test_batch_size = 2
    test_batch, _ = mnist.train.next_batch(test_batch_size)
    processed_batch = preprocess_batch(test_batch)
    if visualize_bool:
        for i in range(test_batch_size):
            image = test_batch[i, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image)
            plt.show()
            image = processed_batch[i, :]
            image = np.reshape(image, [input_shape[0], input_shape[1]])
            plt.imshow(image)
            plt.show()


def parse_batch():
    parse_batch_x, parse_batch_y = mnist.train.next_batch(batch_size)
    return parse_batch_x, parse_batch_y


def dct_and_crop(array):
    output = dct(dct(array.T).T)
    output = output[0:input_shape[0], 0:input_shape[1]]
    return output


def wavelet_and_crop(array):
    wp = pywt.wavedec2(array, 'db1', mode='symmetric', level=2)
    level1 = np.vstack((np.hstack((wp[0], wp[1][0])), np.hstack((wp[1][1], wp[1][2]))))
    level2 = np.vstack((np.hstack((level1, wp[2][0])), np.hstack((wp[2][1], wp[2][2]))))
    output = level2[0:input_shape[0], 0:input_shape[1]]
    return output


cached_psi = None


def shearlet_and_crop(array):
    global cached_psi
    if cached_psi is None:
        ST, cached_psi = shearletTransformSpect(array)
    else:
        ST, psi = shearletTransformSpect(array, cached_psi)
    a = ST[0::4, 0::4, 0]
    h = ST[0::4, 0::4, 1]
    v = ST[0::4, 0::4, 2]
    d = ST[0::4, 0::4, 3]
    level0 = np.block([[a, h], [v, d]])
    output = np.array(level0)
    '''
    cc = ST[0::2,0::2, 0:9]
    psi_cc = psi[0::2,0::2, 0:9]
    output = inverseShearletTransformSpect(cc, psi_cc)
    '''
    return output


def scale_down(array):  # this takes forever to solve
    output = resize(array, (input_shape[0], input_shape[1]), mode='reflect')
    return output


# this replaces preprocess_batch
def tf_scale_down_batch(input_batch):
    output = tf.reshape(input_batch, [input_batch.shape[0], 28, 28, 1])
    output = tf.image.resize_images(output, [input_shape[0], input_shape[1]])
    output = tf.reshape(output, [input_batch.shape[0], input_shape[0]*input_shape[1]])
    return output


def preprocess_image(array):
    # input array shape is [28,28]
    # output shape should be [-1, input_shape[0], input_shape[1], 1]
    output = np.reshape(wavelet_and_crop(array), [-1, input_shape[0]*input_shape[1]])
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


def train_at_points():
    global input_shape
    if gpu_ident == 0:
        point_list = [28, 26, 24]
    if gpu_ident == 1:
        point_list = [22, 20, 18]
    if gpu_ident == 2:
        point_list = [16, 14, 12]
    if gpu_ident == 3:
        point_list = [10, 8]
    file_string = "output_gpu" + str(gpu_ident) + ".txt"
    thefile = open(file_string, 'w')
    thefile.close()
    for point in point_list:
        acc = float(0)
        input_shape = [point, point]
        x = tf.placeholder('float', [None, input_shape[0] * input_shape[1]])
        for i in range(n_evaluations):
            print("Starting Evaluation", i)
            acc = acc + train_network(x)
        acc = acc/n_evaluations
        print("Average Accuracy at point ", point, ": ", acc)
        thefile = open(file_string, 'a')
        thefile.write("%s\n" % point)
        thefile.write("%s\n" % acc)
        thefile.close()


train_at_points()
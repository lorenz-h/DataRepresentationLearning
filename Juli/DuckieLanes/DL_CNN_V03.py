import tensorflow as tf
import numpy as np
from scipy.misc import imsave

batch_size = 16
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.0001
n_epochs = 30


def parse_fn(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_normalized = tf.image.per_image_standardization(img_decoded)
    img_normalized = tf.image.crop_to_bounding_box(img_normalized,offset_height=120,offset_width=0,target_height=360, target_width=640)
    raw = tf.read_file(label_path)

    label = tf.decode_raw(raw, out_type=tf.float32)*100
    noise = tf.random_normal(shape=tf.shape(label), mean=0.0, stddev=1.0, dtype=tf.float32)/1000
    label = label + noise
    return img_normalized, label


def create_dataset(evaluation):
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files("Dataset/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files("Dataset/"+test_train+"/*.txt", shuffle=False)
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 6, 6, 1], strides=[1, 4, 4, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, 360, 640, 3])
    x = tf.cast(x, tf.float32)
    w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))
    b_conv1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    b_conv2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(convolution(conv1, w_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)
    conv2_shape = [conv2.shape[1].value, conv2.shape[2].value, conv2.shape[3].value]
    w_fc = tf.Variable(tf.random_normal([conv2_shape[0] * conv2_shape[1] * conv2_shape[2], 64]))
    b_fc = tf.Variable(tf.random_normal([64]))
    fc = tf.reshape(conv2, [-1, conv2_shape[0] * conv2_shape[1] * conv2_shape[2]])
    fc = tf.matmul(fc, w_fc) + b_fc

    w_out = tf.Variable(tf.random_normal([64, 1]))
    b_out = tf.Variable(tf.random_normal([1]))

    output = tf.matmul(fc, w_out) + b_out
    return output


tr_data = create_dataset(evaluation=False)
val_data = create_dataset(evaluation=True)

iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
next_element = iterator.get_next()
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)
prediction = cnn(next_element[0])
with tf.device('/device:CPU:0'):
    loss = tf.cast(tf.losses.absolute_difference(labels=next_element[1], predictions=prediction), dtype=tf.float64)

optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)


equality = tf.equal(prediction, next_element[1])
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for e in range(n_epochs):
        img, l, _, acc = sess.run([next_element[0], loss, train_op, accuracy])
        if e < 2:
            for i in range(4):
                imsave("test"+str(i)+".png", img[i])
        if e % 2 == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(e, l, acc * 100))

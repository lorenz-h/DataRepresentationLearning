import tensorflow as tf
import os
from scipy.fftpack import dct
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

batch_size = 64
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.01
n_epochs = 10
n_evaluations = 4
logging = True
input_shape = [190, 190]
gpu_id = 3


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, input_shape[0], input_shape[1], 3])

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


def open_label(file_path):
    label_file = open(file_path, "r")
    try:
        label = label_file.read()
    finally:
        label_file.close()
    label = float(label)
    return label


def numpy_preprocess(image):
    red = image[..., 0]
    red = red / 255
    red = red.astype(dtype="float32")
    axes = list(range(red.ndim))
    for ax in axes:
        red = dct(red, type=2, axis=ax, norm='ortho')
    green = image[..., 0]
    green = green / 255
    green = green.astype(dtype="float32")
    axes = list(range(green.ndim))
    for ax in axes:
        green = dct(green, type=2, axis=ax, norm='ortho')
    blue = image[..., 0]
    blue = blue / 255
    blue = blue.astype(dtype="float32")
    axes = list(range(blue.ndim))
    for ax in axes:
        blue = dct(blue, type=2, axis=ax, norm='ortho')
    output = np.dstack((red, green, blue))
    output = output.astype(dtype="float32")
    output = output[0:input_shape[0], 0:input_shape[1]]
    return output


def parse_fn(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_cropped = tf.image.crop_to_bounding_box(img_decoded, offset_height=120,offset_width=0,target_height=360, target_width=640)
    img_cropped = tf.cast(img_cropped, dtype=tf.float32)
    img_processed = tf.py_func(numpy_preprocess, [img_cropped], tf.float32)
    label = tf.py_func(open_label, [label_path], "double")
    return img_processed, label


def create_dataset(evaluation):
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.txt", shuffle=False)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def setup_network():
    tr_data = create_dataset(evaluation=False)
    val_data = create_dataset(evaluation=True)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    validation_init_op = iterator.make_initializer(val_data)

    prediction = cnn(next_features)
    loss = tf.cast(tf.losses.mean_squared_error(labels=next_labels, predictions=prediction), dtype=tf.float32)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)

    delta = tf.abs(prediction - tf.cast(next_labels, dtype=tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.less(delta, 0.4), dtype=tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        if logging:
            gpu_logdir = "gpu" + str(gpu_id)
            train_logdir = os.path.join("logs", gpu_logdir, "train")
            test_logdir = os.path.join("logs", gpu_logdir, "test")
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

        sess.run(init_op)
        i = 0
        for epoch in range(1, n_epochs+1):
            sess.run(training_init_op)
            epoch_loss = 0
            epoch_acc = 0
            batches = 0
            while True:
                try:
                    if logging:
                        summary, lss, _, acc = sess.run([merged, loss, optimizer, accuracy])
                        train_writer.add_summary(summary, i)
                        i += 1
                    else:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                    epoch_loss += lss
                    epoch_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            epoch_loss = epoch_loss/batches
            epoch_acc = epoch_acc/batches

            print("Finished Epoch", epoch, "- Training Loss:", epoch_loss, "- Accuracy:", epoch_acc)

        for ev in range(n_evaluations):
            sess.run(validation_init_op)
            eval_acc = 0
            batches = 0
            while True:
                try:
                    images, acc = sess.run([next_features, loss])
                    if batches < 5:
                        image = images[0, ..., 1]
                        imsave(str(batches)+".png", image)
                    eval_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            print("Evaluation", ev, "done.")
        eval_acc = eval_acc/batches
        print("Average Accuracy over", n_evaluations, "was", eval_acc)


def visualize_preprocessing():
    tr_data = create_dataset(evaluation=False)
    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(training_init_op)
        features = sess.run(next_features)
        image = features[0, ...]
        image = image
        print(np.amin(image))
        print(np.amax(image))
        plt.imshow(image)
        plt.show()


setup_network()
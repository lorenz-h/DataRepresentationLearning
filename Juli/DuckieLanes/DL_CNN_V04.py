import tensorflow as tf
from scipy.misc import imsave
from scipy.fftpack import dct
import numpy as np

batch_size = 64
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.001
n_epochs = 8
n_evaluations = 3


def numpy_preprocess(image):
    red = dct(dct(image[..., 0].T).T)
    green = dct(dct(image[..., 1].T).T)
    blue = dct(dct(image[..., 2].T).T)
    #output = np.dstack((red, blue, green))
    # output = output[0:input_shape[0], 0:input_shape[1]]
    return image


def open_label(file_path):
    label_file = open(file_path, "r")
    label = label_file.read()
    label_file.close()
    label = float(label)
    return label


def parse_fn(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_cropped = tf.image.crop_to_bounding_box(img_decoded, offset_height=120,offset_width=0,target_height=360, target_width=640)
    img_cropped = tf.cast(img_cropped, tf.float32)
    img_normalized = tf.image.per_image_standardization(img_cropped)
    img_processed = tf.py_func(numpy_preprocess, [img_normalized], tf.float32)
    label = tf.py_func(open_label, [label_path], "double")
    return img_processed, label


def flip_parse_fn(image_path, label_path):
    img, label = parse_fn(image_path, label_path)
    img = tf.image.flip_left_right(img)
    return img, label


def create_dataset(evaluation):
    global n_epochs
    global n_evaluations
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.txt", shuffle=False)
    normal = tf.data.Dataset.zip((images, labels))
    normal = normal.shuffle(buffer_size=shuffle_buffer_size)
    flipped = normal.take(-1)
    normal = normal.map(map_func=parse_fn)
    flipped = flipped.map(map_func=flip_parse_fn)
    dataset = normal.concatenate(flipped)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, 360, 640, 3])

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
next_batch = iterator.get_next()
training_init_op = iterator.make_initializer(tr_data)
validation_init_op = iterator.make_initializer(val_data)

prediction = cnn(next_batch[0])
loss = tf.cast(tf.losses.absolute_difference(labels=next_batch[1], predictions=prediction), dtype=tf.float32)
optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)

delta = prediction-tf.cast(next_batch[1], dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.less(delta, 0.05), dtype=tf.float32))

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)
    print("-" * 20, "Starting Training", "-" * 20)
    # --------------TRAINING-----------------
    for e in range(n_epochs):
        i = 0
        sess.run(training_init_op)
        while True:
            try:
                i = i+1
                batch = sess.run(next_batch)
                lss, _, acc = sess.run([loss, optimizer, accuracy])
                if i % 4 == 0:
                    print("Processed Batch", i, "with accuracy:", acc)
            except tf.errors.OutOfRangeError:
                print("Epoch", e+1, "out of", n_epochs, "done.", i, "Batches in this epoch, last accuracy:", acc)
                break
    print("-"*20, "Starting Validation", "-"*20)
    # --------------Validation-----------------
    avg_acc = 0
    i = 0
    for e in range(n_evaluations):
        sess.run(validation_init_op)
        while True:
            try:
                lss, _, acc = sess.run([loss, optimizer, accuracy])
                avg_acc += acc
                i += 1
            except tf.errors.OutOfRangeError:
                print("Evaluation", e + 1, "out of", n_evaluations, "done, last accuracys:", acc)
                break
    avg_acc = avg_acc/i
    print(avg_acc)


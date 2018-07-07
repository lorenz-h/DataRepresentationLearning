import tensorflow as tf
import pickle

n_training_samples = 10
n_val_samples = 10
batch_size = 3
keep_rate = 0.9
adam_learning_rate = 0.00001
n_epochs = 20
n_evaluations = 5


def create_placeholder_dataset(offset, n_samples, eval_bool):
    data_paths = []
    if eval_bool == 0:
        subfolder = "Training"
    else:
        subfolder = "Testing"
    for i in range(offset, n_samples):
        data_paths.append("Dataset/"+subfolder+"/sample"+str(i)+".png")
    imgs = tf.constant(data_paths)
    if eval_bool:
        filepath = "Dataset/test_labels.pkl"
    else:
        filepath = "Dataset/train_labels.pkl"
    pkl_file = open(filepath, 'rb')
    loaded_labels = pickle.load(pkl_file)
    labels = loaded_labels[offset:n_samples]
    return imgs, labels


def input_parser(file_path, label):
    img_file = tf.read_file(file_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    return img_decoded, label


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, 480, 640, 3])
    x = tf.cast(x, tf.float32)
    w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 64]))
    b_conv1 = tf.Variable(tf.random_normal([64]))
    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    w_conv2 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
    b_conv2 = tf.Variable(tf.random_normal([128]))
    conv2 = tf.nn.relu(convolution(conv1, w_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)
    conv2_shape = [conv2.shape[1].value, conv2.shape[2].value, conv2.shape[3].value]
    w_fc = tf.Variable(tf.random_normal([conv2_shape[0] * conv2_shape[1] * conv2_shape[2], 128]))
    b_fc = tf.Variable(tf.random_normal([128]))
    fc = tf.reshape(conv2, [-1, conv2_shape[0] * conv2_shape[1] * conv2_shape[2]])
    fc = tf.matmul(fc, w_fc) + b_fc

    w_out = tf.Variable(tf.random_normal([128, 1]))
    b_out = tf.Variable(tf.random_normal([1]))

    output = tf.matmul(fc, w_out) + b_out
    return output


def setup_network():
    # create Placeholder Datasets
    train_imgs, train_labels = create_placeholder_dataset(0, n_training_samples, 0)
    val_imgs, val_labels = create_placeholder_dataset(0, n_val_samples, 1)

    # create TensorFlow Dataset objects
    tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    tr_data = tr_data.map(input_parser).batch(batch_size)
    tr_data = tr_data.prefetch(3*batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
    val_data = val_data.map(input_parser).batch(10)
    val_data = val_data.prefetch(3 * 10)
    # create TensorFlow Iterator object
    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_element = iterator.get_next()

    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(tr_data)
    validation_init_op = iterator.make_initializer(val_data)
    prediction = cnn(next_element[0])
    next_label_reshaped = tf.cast(tf.expand_dims(next_element[1], -1), tf.float32)
    loss = tf.losses.mean_squared_error(labels=next_label_reshaped, predictions=prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)
    equality = tf.equal(prediction, next_label_reshaped)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(init_op)
        sess.run(training_init_op)
        for i in range(n_epochs):
            l, _, acc = sess.run([loss, optimizer, accuracy])
            if i % 20 == 0:
                print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
        sess.run(validation_init_op)
        avg_acc = 0
        for i in range(n_evaluations):
            acc = sess.run([accuracy])
            avg_acc += acc[0]


setup_network()
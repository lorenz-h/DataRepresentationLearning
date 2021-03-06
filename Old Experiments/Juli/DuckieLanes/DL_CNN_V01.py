import tensorflow as tf
import argparse
import os
from import_from_txt import parse_dataset

########################################################################################################################
# hyperparameter defaults(some can be ovverriden through arguments):
gpu_id = 6
batch_size = 128
n_classes = 10
keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32)
n_epochs = 399
n_evaluations = 3
adam_learning_rate = 0.001
logging = False
########################################################################################################################


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):

    x = tf.reshape(x, shape=[-1, 480, 640, 3])

    w_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 64]))
    b_conv1 = tf.Variable(tf.random_normal([64]))
    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    w_conv2 = tf.Variable(tf.random_normal([5, 5, 64, 128]))
    b_conv2 = tf.Variable(tf.random_normal([128]))
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


def setup_network():
    print("Parsing Dataset...")
    training_data, training_labels, validation_images, validation_labels = parse_dataset()
    print("Done.")
    dx_train = tf.data.Dataset.from_tensor_slices(training_data)
    dy_train = tf.data.Dataset.from_tensor_slices(training_labels).map(lambda z: tf.one_hot(z, 10))
    train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(batch_size)

    dx_valid = tf.data.Dataset.from_tensor_slices(validation_images)
    dy_valid = tf.data.Dataset.from_tensor_slices(validation_labels).map(lambda z: tf.one_hot(z, 10))
    valid_dataset = tf.data.Dataset.zip((dx_valid, dy_valid)).shuffle(500).repeat().batch(100)
    print("created Datasets")
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    # define operations that will switch the iterator from one part of the dataset to the other
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)

    logits = cnn(next_element[0])
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)
    # get accuracy
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    init_op = tf.global_variables_initializer()

    def train_network():
        # run the training
        with tf.Session() as sess:
            if logging:
                gpu_logdir = "gpu"+str(gpu_id)
                train_logdir = os.path.join("logs", gpu_logdir, "train")
                test_logdir = os.path.join("logs", gpu_logdir, "test")
                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
                test_writer = tf.summary.FileWriter(test_logdir)
            else:
                merged = tf.Variable(1)

            sess.run(init_op)
            sess.run(training_init_op)
            for i in range(n_epochs):

                if logging:
                    summary, l, _, acc = sess.run([merged, loss, optimizer, accuracy])
                    train_writer.add_summary(summary, i)
                else:
                    l, _, acc = sess.run([loss, optimizer, accuracy])
                if i % 20 == 0:
                    print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
            # now setup the validation run
            # re-initialize the iterator, but this time with validation data
            sess.run(validation_init_op)
            avg_acc = 0
            for i in range(n_evaluations):
                acc = sess.run([accuracy])
                avg_acc += acc[0]
            print("Average validation set accuracy over {} iterations is {:.2f}%".format(n_evaluations, (
                        avg_acc / n_evaluations) * 100))
            return avg_acc

    train_network()


class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=type(globals()["gpu_id"]))
    parser.add_argument("--batch_size", type=type(globals()["batch_size"]))
    parser.add_argument("--adam_learning_rate", type=type(globals()["adam_learning_rate"]))
    parser.add_argument("--keep_rate", type=type(globals()["keep_rate"]))
    parser.add_argument("--n_epochs", type=type(globals()["n_epochs"]))
    parser.add_argument("--logging", type=type(globals()["logging"]))

    args = parser.parse_args()

    print("#"*100, "\n")
    print("Arguments:")
    for argument in args.__dict__:
        if args.__dict__[argument] is None:
            print(Colors.WARNING, argument, "argument not given... Falling back to default value of: ",
                  globals()[argument], Colors.ENDC)
        else:
            globals()[argument] = getattr(args, argument)
            print(Colors.OKGREEN, argument, "argument parsed successfully as", globals()[argument], Colors.ENDC)

    print("")
    print("#" * 100)


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_network()


if __name__ == "__main__":
    main()

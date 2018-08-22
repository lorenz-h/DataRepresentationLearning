import tensorflow as tf
import argparse
import os
from scipy.misc import imsave, imread
import subprocess
import numpy as np
import matplotlib.pyplot as plt

batch_size = 16
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.01
n_epochs = 30
n_evaluations = 5
logging = True

gpu_id = 0
input_shape = (2, 640)
network_depth = 1

dataset_folder = "Shearlet_Dataset"


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def cnn(x):
    if len(input_shape) == 3:
        depth = 3
    else:
        depth = 1
    input_layer = tf.reshape(x, [-1, input_shape[0], input_shape[1], depth])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=None)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=None)
    dense_2 = tf.layers.dense(inputs=dense, units=64, activation=None)
    out = tf.layers.dense(inputs=dense_2, units=1, activation=None)
    return out


def grab_files(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)
    print(img_decoded.shape)
    img_decoded = tf.cast(img_decoded / 255, dtype=tf.float32)
    label_file = tf.read_file(label_path)
    label = tf.string_to_number(label_file, out_type=tf.float64)
    label = tf.cast(label * 100.0, dtype=tf.float32)
    return img_decoded, label


def normal_parse_fn(image_path, label_path):
    image, label = grab_files(image_path, label_path)
    return image, label


def flipped_parse_fn(image_path, label_path):
    image, label = grab_files(image_path, label_path)
    image = tf.image.flip_left_right(image)
    return image, label


def create_dataset(evaluation):
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files(dataset_folder+"/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files(dataset_folder+"/"+test_train+"/*.txt", shuffle=False)
    normal = tf.data.Dataset.zip((images, labels))
    flipped = normal.take(-1)

    normal = normal.map(map_func=normal_parse_fn)
    flipped = flipped.map(map_func=flipped_parse_fn)
    dataset = normal.concatenate(flipped)

    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def setup_network():
    tr_data = create_dataset(evaluation=False)
    eval_data = create_dataset(evaluation=True)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    eval_init_op = iterator.make_initializer(eval_data)

    prediction = cnn(next_features)

    loss = tf.reduce_mean(tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE))

    loss_summary = tf.summary.scalar('loss', loss)

    batch_avg_labels = tf.reduce_mean(tf.abs(next_labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)
    accuracy = tf.cast(tf.losses.absolute_difference(labels=tf.expand_dims(next_labels, -1), predictions=prediction),
                       dtype=tf.float32)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        if logging:
            sub_logdir = "gpu" + str(batch_size)+"_"+str(adam_learning_rate)+"_"+str(n_epochs)
            train_logdir = os.path.join("logs", sub_logdir, "train")
            test_logdir = os.path.join("logs", sub_logdir, "test")
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

        sess.run(init_op)
        print("Starting Training...")
        i = 0
        for epoch in range(1, n_epochs+1):
            sess.run(training_init_op)
            avg_epoch_loss = 0
            batch_counter = 0
            while True:  # iterate over all batches
                try:
                    if logging:
                        lss, summary, _, acc = sess.run([loss, loss_summary, optimizer, accuracy])
                        train_writer.add_summary(summary, i)
                    else:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                    avg_epoch_loss += lss
                    batch_counter += 1
                    i += 1
                except tf.errors.OutOfRangeError:
                    break
            avg_epoch_loss = avg_epoch_loss / batch_counter

            print("Finished Epoch", epoch, "- Training Loss:", avg_epoch_loss)

        print("Starting Evaluation...")
        batch_counter = 0
        avg_eval_loss = 0
        avg_labels = 0

        for ev in range(1, n_evaluations+1):
            sess.run(eval_init_op)
            avg_labels = 0
            while True:
                try:
                    if logging:
                        lss, summary, labels = sess.run([loss, loss_summary, batch_avg_labels])
                        test_writer.add_summary(summary, batch_counter)
                    else:
                        lss, labels = sess.run([loss, batch_avg_labels])
                    avg_eval_loss += lss
                    avg_labels += labels
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    break
            print("Evaluation", ev,"done.")
        avg_eval_loss = avg_eval_loss / batch_counter
        avg_labels = avg_labels / batch_counter
        print("Average Loss:", avg_eval_loss, "Average Label Size:", avg_labels)


class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def str2bool(v):
    # this is a neccesary workaround to ensure that bools will be parsed correctly
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=type(globals()["gpu_id"]))
    parser.add_argument("--batch_size", type=type(globals()["batch_size"]))
    parser.add_argument("--adam_learning_rate", type=type(globals()["adam_learning_rate"]))
    parser.add_argument("--n_epochs", type=type(globals()["n_epochs"]))
    parser.add_argument("--logging", type=str2bool)

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


def check_available_gpus():
    system_gpus = 7
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            return gpu
    assert False, "All GPUs are currently busy."


def get_input_shape():
    file_string = dataset_folder + "/Training/sample0.png"
    image = imread(file_string)
    # plt.imshow(image)
    # plt.show()
    return image.shape


def main():
    global gpu_id
    global input_shape

    gpu_id = check_available_gpus()
    print("GPU", gpu_id, "is free and will be used.")

    input_shape = get_input_shape()
    print("Found input shape to be", input_shape)

    parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_network()


if __name__ == "__main__":
    main()


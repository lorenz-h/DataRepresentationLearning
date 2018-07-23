import tensorflow as tf
import argparse
import os
from scipy.misc import imsave, imread
import subprocess
import numpy as np


batch_size = 64
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.01
n_epochs = 3
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
    input_layer = tf.reshape(x, [-1, input_shape[0], input_shape[1], 1])
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
    val_data = create_dataset(evaluation=True)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    validation_init_op = iterator.make_initializer(val_data)

    prediction = cnn(next_features)

    loss = tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE)

    avg_loss = tf.reduce_mean(loss)
    stddev_loss = tf.sqrt(tf.reduce_mean(tf.square(loss - avg_loss)))
    avg_labels = tf.reduce_mean(next_labels)
    stddev_labels = tf.sqrt(tf.reduce_mean(tf.square(next_labels - avg_labels)))

    tf.summary.scalar('stddev_labels', stddev_labels)
    avg_loss_summary = tf.summary.scalar('avg_loss', avg_loss)
    tf.summary.scalar('stddev_loss', stddev_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(avg_loss)

    accuracy = tf.cast(tf.losses.absolute_difference(labels=tf.expand_dims(next_labels, -1), predictions=prediction), dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        if logging:
            sub_logdir = "gpu" + str(batch_size)+"_"+str(adam_learning_rate)+"_"+str(n_epochs)
            train_logdir = os.path.join("logs", sub_logdir, "train")
            test_logdir = os.path.join("logs", sub_logdir, "test")
            batchwise_summary = tf.summary.merge([avg_loss_summary])
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

        sess.run(init_op)
        i = 0
        print("Starting Training...")
        for epoch in range(1, n_epochs + 1):
            sess.run(training_init_op)
            epoch_loss.assign
            epoch_acc = 0
            batches = 0
            while True:
                try:
                    if logging:
                        summary, lss, _, acc = sess.run([batchwise_summary, avg_loss, optimizer, accuracy])
                        train_writer.add_summary(summary, i)  # the loss will be logged for every batch
                        i += 1
                    else:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                    epoch_loss += lss
                    epoch_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            epoch_loss = epoch_loss / batches
            epoch_acc = epoch_acc / batches

            train_writer.add_summary(epoch_loss, epoch)
            train_writer.add_summary(epoch_acc, epoch)
            print("Finished Epoch", epoch, "- Training Loss:", epoch_loss, "- Accuracy:", epoch_acc)
        print("Starting Evaluation...")
        avg_acc = 0
        for ev in range(n_evaluations):
            sess.run(validation_init_op)
            eval_acc = 0
            batches = 0
            avg_label = 0
            while True:
                try:
                    summary, images, labels, acc = sess.run([avg_loss, next_features, next_labels, avg_loss])
                    test_writer.add_summary(summary, ev)
                    avg_label += np.mean(np.absolute(labels))
                    if batches < 5:
                        image = images[1, ...]
                        image = np.reshape(image, [images.shape[1], images.shape[2]])
                        imsave(str(batches) + ".png", image)
                    eval_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            avg_label = avg_label / batches
            print("Evaluation", ev, "done.")
            eval_acc = eval_acc / batches
            avg_acc += eval_acc
        avg_acc = avg_acc / n_evaluations
        print("The average label size was", avg_label)
        print("Average absolute error over", n_evaluations, "evaluations was", avg_acc)


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
    return image.shape


def main():
    global gpu_id
    global input_shape

    # gpu_id = check_available_gpus()
    print("GPU", gpu_id, "is free and will be used.")

    input_shape = get_input_shape()
    print("Found input shape to be", input_shape)

    parse_arguments()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_network()


if __name__ == "__main__":
    main()


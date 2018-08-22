import tensorflow as tf
import os
import argparse

from DL_Utilities import check_available_gpus, get_input_shape, Colors, str2bool, Logger
from DL_Input_Pipeline import create_dataset


batch_size = 32
n_epochs = 9
n_evaluations = 4
learning_rate = 0.006090948016583396
logging = False
prefetch_buffer_size = batch_size*2
gpu_id = 0

dataset_folder = "Shearlet_Dataset_V02"
n_runs = 2
allowed_arguments = ["n_epochs", "logging", "n_evaluations", "learning_rate",
                     "dataset_folder", "n_runs", "batch_size", "gpu_id"]


def parse_arguments(allowed_args):
    parser = argparse.ArgumentParser()
    for arg in allowed_args:
        if isinstance(globals()[arg], bool):
            parser.add_argument("--"+arg, type=str2bool)
        else:
            parser.add_argument("--" + arg, type=type(globals()[arg]))
    args = parser.parse_args()
    print("")
    print("Arguments:")
    for argument in args.__dict__:
        if args.__dict__[argument] is None:
            print(Colors.WARNING, argument, "argument not given... Falling back to default value of: ",
                  globals()[argument], Colors.ENDC)
        else:
            globals()[argument] = getattr(args, argument)
            print(Colors.OKGREEN, argument, "argument parsed successfully as", globals()[argument], Colors.ENDC)

    print("")


def cnn(x, input_shape):
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


def setup_network(input_shape, dpth):
    tr_data = create_dataset(False, batch_size, dataset_folder, dpth)
    eval_data = create_dataset(True, batch_size, dataset_folder, dpth)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    eval_init_op = iterator.make_initializer(eval_data)

    prediction = cnn(next_features, input_shape)

    loss = tf.reduce_mean(tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    batch_avg_labels = tf.reduce_mean(tf.abs(next_labels))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        def training(lg_dir):
            training_logger = Logger(lg_dir+"/train")
            epoch_loss = 0
            for ep in range(1, n_epochs + 1):
                sess.run(training_init_op)
                batches = 0
                while True:
                    try:
                        lss, _ = sess.run([loss, optimizer])
                        epoch_loss = epoch_loss + lss
                        batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_loss = epoch_loss / batches
                print("Epoch", ep, "done. Loss:", epoch_loss)
                if logging:
                    training_logger.log_scalar("epoch_loss", epoch_loss, ep)

        def evaluation(lg_dir):
            eval_logger = Logger(lg_dir+"/eval")
            batches = 0
            avg_labels = 0
            avg_loss = 0
            for ev in range(1, n_evaluations+1):
                sess.run(eval_init_op)
                while True:
                    try:
                        b_avg_lab, images, lss = sess.run([batch_avg_labels, next_features, loss])
                        avg_labels += b_avg_lab
                        avg_loss += lss
                        batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                eval_logger.log_image("img", images, ev, sess)

                print("Evaluation", ev, "done. Loss:", lss)
            avg_loss = avg_loss / batches
            avg_labels = avg_labels / batches
            if logging:
                eval_logger.log_scalar("avg_loss", avg_loss, ev)
                eval_logger.log_scalar("rel_error", avg_loss/avg_labels, ev)
            return avg_loss, avg_labels

        experiment_logdir = "gpu" + str(batch_size) + "_" + str(learning_rate) + "_" \
                            + str(n_epochs) + "_" + dataset_folder
        logdir = os.path.join("logs", experiment_logdir)

        training(logdir)
        return evaluation(logdir)


def standalone():
    global depth
    global gpu_id
    gpu_id = check_available_gpus()
    print("#" * 100, "\n")
    print(Colors.OKGREEN, "GPU", gpu_id, "is free and will be used.", Colors.ENDC)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    parse_arguments(allowed_arguments)
    input_shape = get_input_shape(dataset_folder)
    print(Colors.OKBLUE, "Found input shape to be", input_shape, Colors.ENDC)
    print("#" * 100)
    if len(input_shape) == 3:
        depth = 3
    else:
        depth = 1

    acc = 0
    label_size = 0
    for it in range(0, n_runs):
        a, ls = setup_network(input_shape, depth)
        acc += a
        label_size += ls
        tf.reset_default_graph()
    acc = acc/n_runs
    label_size = label_size/n_runs
    print("Average Loss was:", acc, "for labels of size:", label_size)
    try:
        file_path = "logs/" + dataset_folder + "_" + str(n_epochs) \
                    + "_" + str(n_evaluations) + "_" + str(learning_rate)+".txt"
        label_file = open(file_path, "w")
        label_file.write(str(acc))
        label_file.write(str(label_size))
    finally:
        label_file.close()


def child():
    global depth
    global gpu_id
    print("#" * 100, "\n")

    parse_arguments(allowed_arguments)

    print(Colors.OKGREEN, "GPU", gpu_id, "has been parsed and will be used.", Colors.ENDC)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    input_shape = get_input_shape(dataset_folder)
    print(Colors.OKBLUE, "Found input shape to be", input_shape, Colors.ENDC)
    print("#" * 100)
    if len(input_shape) == 3:
        depth = 3
    else:
        depth = 1

    acc = 0
    label_size = 0
    for it in range(0, n_runs):
        a, ls = setup_network(input_shape, depth)
        acc += a
        label_size += ls
        tf.reset_default_graph()
    acc = acc / n_runs
    print("Average Loss was: AxzcodeK" + str(acc) + "AxzcodeK for labels of size:", label_size)


def main():
    child()


if __name__ == "__main__":
    main()

import tensorflow as tf
import os
import argparse

from DL_Utilities import check_available_gpus, get_input_shape, Colors, str2bool, Logger, get_convolutions
from DL_Input_Pipeline import create_dataset

# #############################################################################
batch_size = 32
n_epochs = 5
n_evaluations = 3
learning_rate = 0.001
logging = False
prefetch_buffer_size = batch_size*2
gpu_id = 1
n_convolutions = 2
n_dense_nodes = 196
n_runs = 1

# #############################################################################
dataset_folder = "Shearlet_Dataset_V02"
allowed_arguments = ["n_epochs", "logging", "n_evaluations", "learning_rate",
                     "dataset_folder", "n_runs", "batch_size", "gpu_id", "n_convolutions", "n_dense_nodes"]


def parse_arguments(allowed_args):
    """
    uses argparse to capture the arguments parsed to the script and changes the values of the global variables specified
    in allowed_args
    :param allowed_args: a list of global variables which contents can be changed by parsing them.
    :return: ---
    """
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


def cnn(x, input_shape, convs):
    """
    :param x: a single image or a batch of images to be passed into the network. Tensorflow automatically adjusts wether
    one image or a batch is passed.
    :param input_shape: the shape of the input images (x,y,#channels)
    :param convs: a list of convolutions the network should use of shape (conv_depth, kernel_size_x, kernel_size_y )
    :return: the output of the network
    """
    input_layer = tf.reshape(x, [-1, input_shape[0], input_shape[1], input_shape[2]])
    input_arr = input_layer
    for conv in convs:
        cnv = tf.layers.conv2d(
            inputs=input_arr,
            filters=conv[0],
            kernel_size=[conv[1], conv[2]],
            padding="same",
            activation=None)
        input_arr = tf.layers.max_pooling2d(inputs=cnv, pool_size=[2, 2], strides=2)
    pool2_flat = tf.layers.flatten(input_arr)
    dense = tf.layers.dense(inputs=pool2_flat, units=n_dense_nodes, activation=None)
    dense_2 = tf.layers.dense(inputs=dense, units=int(n_dense_nodes/2), activation=None)
    out = tf.layers.dense(inputs=dense_2, units=1, activation=None)
    return out


def setup_network(input_shape, convs):
    """
    this sets up one training and validation cycle for the network.
    :param input_shape: the shape of the input dataset
    :param convs: the convolutions the network should use
    :return: tuple of average loss and average labels of the evaluations
    """
    tr_data = create_dataset(False, batch_size, dataset_folder, input_shape)
    eval_data = create_dataset(True, batch_size, dataset_folder, input_shape)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    eval_init_op = iterator.make_initializer(eval_data)

    prediction = cnn(next_features, input_shape, convs)

    element_loss = tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE)

    loss = tf.reduce_mean(element_loss)

    accuracy = tf.reduce_mean(tf.cast(tf.less(element_loss, 0.63), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    batch_avg_labels = tf.reduce_mean(tf.abs(next_labels))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        def training(lg_dir):
            """
            This starts training the network and logs the results (if logging==true) to the directory lodir
            :param lg_dir: the directory to store logs in
            :return: ---
            """
            training_logger = Logger(lg_dir+"/train")
            epoch_loss = 0
            for ep in range(1, n_epochs + 1):
                sess.run(training_init_op)
                batches = 0
                while True:
                    try:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                        epoch_loss = epoch_loss + lss
                        batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_loss = epoch_loss / batches
                print("Epoch", ep, "done. Loss:", epoch_loss)
                print("last accuracy", acc)
                if logging:
                    training_logger.log_scalar("epoch_loss", epoch_loss, ep)

        def evaluation(lg_dir):
            """
            This starts n_evaluations evaluations of the network and logs the results (if logging==true) to the directory lodir
            :param lg_dir: the directory to store logs in
            :return: a tuple of the avg_loss and avg_labels over all evaluations of the network
            """
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
                if logging:
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
    """
    This should be called when the training and evaluation of the network is called by hand. Looks for an available GPU
    and places the network on that gpu. Parses the arguments and finds the input shape. Then calls setup_network n_runs
    times to average out the varying accuracys caused by the random initialization of the network
    :return: ---
    """
    global gpu_id
    gpu_id = check_available_gpus()
    print("#" * 100, "\n")
    print("RUNNING STANDALONE VERSION")
    print(Colors.OKGREEN, "GPU", gpu_id, "is free and will be used.", Colors.ENDC)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    parse_arguments(allowed_arguments)
    input_shape = get_input_shape(dataset_folder)
    print(Colors.OKBLUE, "Found input shape to be", input_shape, Colors.ENDC)
    print("#" * 100)

    acc = 0
    label_size = 0
    convolutions = get_convolutions(n_convolutions)
    for it in range(0, n_runs):
        a, ls = setup_network(input_shape, convolutions)
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
    """
    this should be called only by DL_Optimizer.py. Parses the arguments and finds the input shape. Then calls
    setup_network n_runs times to average out the varying accuracys caused by the random initialization of the network
    :return: ---
    """
    global gpu_id
    print("#" * 100, "\n")

    parse_arguments(allowed_arguments)

    print(Colors.OKGREEN, "GPU", gpu_id, "has been parsed and will be used.", Colors.ENDC)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    input_shape = get_input_shape(dataset_folder)
    print(Colors.OKBLUE, "Found input shape to be", input_shape, Colors.ENDC)
    print("#" * 100)

    acc = 0
    label_size = 0
    convolutions = get_convolutions(n_convolutions)
    for it in range(0, n_runs):
        a, lss = setup_network(input_shape, convolutions)
        acc += a
        label_size += lss
        tf.reset_default_graph()  # this clears the graph
    acc = acc / n_runs
    print("Average Loss was: AxzcodeK" + str(acc) + "AxzcodeK for labels of size:", label_size)


def main():
    child()
    tf.reset_default_graph()


if __name__ == "__main__":
    main()

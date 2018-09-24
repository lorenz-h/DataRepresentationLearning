#!/usr/bin/env python
"""
This file contains all the definitions for a single Network instance. Its setup_process function gets called
by the Optimizer from ID_Asynchronous_Optimizer.
"""

import tensorflow as tf
import os
import subprocess

from _utils.ID_utils import Logger, ParameterBatch, c_print
from ID_Input_Pipeline import create_dataset


def cnn(x, args):
    input_layer = tf.reshape(x, [-1, args.input_shape[0], args.input_shape[1], args.input_shape[2]])
    input_arr = input_layer

    def conv_layer(conv_input, convolution):
        cnv = tf.layers.conv2d(
            inputs=conv_input,
            filters=convolution[0],
            kernel_size=[convolution[1], convolution[2]],
            padding="same",
            activation=tf.nn.relu,
            name="conv" + str(convolution[0]))
        new_input_arr = tf.layers.max_pooling2d(inputs=cnv, pool_size=[3, 3],
                                                strides=2, name="pool_2d" + str(convolution[0]))
        return new_input_arr

    for conv in args.convolutions:
        input_arr = conv_layer(input_arr, conv)

    pool2_flat = tf.layers.flatten(input_arr, name="FlattenBeforeDense")
    dense = tf.layers.dense(inputs=pool2_flat, units=args.n_dense_nodes, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense, units=int(args.n_dense_nodes/2), activation=tf.nn.relu)
    out = tf.layers.dense(inputs=dense_2, units=1, activation=None)
    return out


def fnn(x, args):
    input_layer = tf.reshape(x, [-1, args.input_shape[0], args.input_shape[1], args.input_shape[2]])
    flattened = tf.layers.flatten(input_layer, name="flatten_image")
    input_tensor = flattened

    def dense_layer(d_input, index):
        dense = tf.layers.dense(inputs=d_input,
                                units=int(args.n_dense_nodes / (index*args.dense_size_convergence + 1)),
                                activation=tf.nn.relu, name="dense_" + str(index))
        return dense

    for i in range(args.n_dense_layers):
        input_tensor = dense_layer(input_tensor, i)
    dense_no_activation = tf.layers.dense(inputs=input_tensor, units=20, activation=None, name="continuation_layer")
    out = tf.layers.dense(inputs=dense_no_activation, units=1, activation=None, name="output_node")
    return out


def spawn_network(args):
    """
    Defines the graph and runs the network.
    :param args: ParameterBatch object as defined in ID_Optimizer.py
    :return: either test or eval loss, depending on args.training
    """
    tf.reset_default_graph()
    train_data = create_dataset(args.batch_size, args.train_csv_file)
    eval_data = create_dataset(args.batch_size, args.eval_csv_file)
    test_data = create_dataset(args.batch_size, args.test_csv_file)

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)
    test_init_op = iterator.make_initializer(test_data)

    if args.use_conv_net:
        prediction = cnn(next_features, args)
    else:
        prediction = fnn(next_features, args)

    element_loss = tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE)

    loss = tf.reduce_mean(element_loss)
    static_approx_error = tf.cast(0.13651795418710108, dtype=tf.float32)

    accuracy = tf.divide((static_approx_error - loss), static_approx_error)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)
    with tf.Session() as sess:
        tf.RunOptions(report_tensor_allocations_upon_oom=True),
        sess.run(tf.global_variables_initializer())

        def test_performance():
            """
            tests the networks performance.
            :return: the training accuracy
            """
            test_logger = Logger(logdir + "/test")
            sess.run(test_init_op)
            batches = 0
            test_loss = 0
            while True:
                try:
                    lss, acc, images = sess.run([loss, accuracy, next_features])
                    test_loss += lss
                    batches += 1
                    if batches % 50 == 0:
                        print("GPU" + str(args.gpu_id) + ": Testbatch " + str(batches) + " done.")
                except tf.errors.OutOfRangeError:
                    break
            assert batches is not 0, "NO TEST DATA PASSED"
            test_logger.log_image(tag="testing_image", images=images, step=0, sess=sess)
            test_loss /= batches
            return test_loss

        def evaluate_performance(ep):
            """
            runs the network on the evaluation_dataset.
            :param ep: the training epoch in which the evaluation is performed
            :return: the evaluation loss
            """
            sess.run(eval_init_op)
            evaluation_logger = Logger(logdir + "/eval")
            batches = 0
            ev_loss = 0
            ev_acc = 0
            while True:
                try:
                    lss, acc = sess.run([loss, accuracy])
                    ev_loss = ev_loss + lss
                    ev_acc = ev_acc + acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            assert batches is not 0, "NO EVALUATION DATA PASSED"
            ev_loss /= batches
            ev_acc /= batches
            evaluation_logger.log_scalar("eval_training_loss", ev_loss, ep)
            evaluation_logger.log_scalar("eval_training_acc", ev_acc, ep)
            return ev_loss, ev_acc

        def training():
            """
            trains the network and evaluates the performance every 3 epochs. Once eval error starts increasing
            stop the training and return the smallest evaluation error achieved.
            :return: the smallest evaluation loss.
            """
            training_logger = Logger(logdir+"/train")
            eval_losses = []
            epoch = 0
            while epoch < args.n_max_epochs:
                epoch_loss = 0
                epoch_acc = 0
                batches = 0
                sess.run(training_init_op)
                while True:
                    try:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                        epoch_loss += lss
                        epoch_acc += acc
                        batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_loss /= batches
                epoch_acc /= batches
                training_logger.log_scalar("epoch_training_loss", epoch_loss, epoch)
                training_logger.log_scalar("epoch_training_acc", epoch_acc, epoch)
                if epoch % 3 == 1:
                    eval_loss, eval_acc = evaluate_performance(epoch)
                    training_logger.log_scalar("epoch_eval_loss", eval_loss, epoch)
                    training_logger.log_scalar("epoch_eval_acc", eval_acc, epoch)
                    eval_losses.append(eval_loss)
                    n_better_setups = 0
                    for stored_lss in eval_losses:
                        if stored_lss < eval_loss:
                            n_better_setups += 1
                    if n_better_setups > 2:
                        print("GPU" + str(args.gpu_id)+" - eval error has been increasing again")
                        break
                    if epoch > int(args.n_max_epochs / 2):
                        if eval_acc < 0.1:
                            break
                epoch += 1
            smallest_eval_loss = sorted(eval_losses)[0]
            print("GPU"+str(args.gpu_id)+" - finished training.")
            return smallest_eval_loss

        if args.use_conv_net:
            experiment_logdir = "gpu" + str(args.gpu_id) + "_" + str(args.learning_rate) + \
                                str(args.n_dense_nodes) + "_" + str(len(args.convolutions)) + str(args.testing)
        else:
            experiment_logdir = "gpu" + str(args.gpu_id) + "_" + str(args.learning_rate) + \
                                str(args.n_dense_nodes) + "_" + str(args.dense_size_convergence) + \
                                "_" + str(args.n_dense_layers) + str(args.testing)
        logdir = os.path.join("_logs", experiment_logdir)
        if args.testing:
            training()
            return test_performance()
        else:
            return training()


def setup_process_environment(args):
    """
    creates an environment in which to run the network. spawns the network n_runs times and averages the loss returned
    by the network ( either eval or test, depending on args.training)
    :param args: ParameterBatch object as defined in ID_Optimizer.py
    :return: ---
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    acc = 0
    for it in range(0, args.n_runs):
        lss = spawn_network(args)
        acc += lss
        tf.reset_default_graph()  # this clears the graph
    acc = acc / args.n_runs
    return acc


def main():
    c_print("Running Standalone Version using default parameters.", color="blue")
    command_str = "(rm -r _logs)"
    subprocess.run(command_str, shell=True)
    params = ParameterBatch(gpu_id=0, training=False)
    setup_process_environment(params)


if __name__ == "__main__":
    main()


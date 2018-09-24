#!/usr/bin/env python
"""
This file contains all the definitions for a single Network instance. Its spawn_network function gets called
by the Optimizer. It is also possible to run a standalone network that will fall back to the default parameters
specified in _utils.ID_utils
"""

import tensorflow as tf
import os
import subprocess

from _utils.ID_utils import Logger, ParameterBatch, generate_logdir, check_available_gpus
from ID_Input_Pipeline import create_dataset


def cnn(x, params):
    """
    creates the tensorflow graph for a single convolutional neural network.
    :param x: input tensor for the network with shape as defined in params.input_shape
    :param params: ParameterBatch object containing all hyperparameters.
    :return: output tensor for the network with shape batch_size x 1.
    """
    input_layer = tf.reshape(x, [-1, params.input_shape[0], params.input_shape[1], params.input_shape[2]])
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

    for conv in params.convolutions:
        input_arr = conv_layer(input_arr, conv)

    pool2_flat = tf.layers.flatten(input_arr, name="FlattenBeforeDense")
    dense = tf.layers.dense(inputs=pool2_flat, units=params.n_dense_nodes, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense, units=int(params.n_dense_nodes/2), activation=tf.nn.relu)
    out = tf.layers.dense(inputs=dense_2, units=1, activation=None)
    return out


def fnn(x, params):
    """
    creates the tensorflow graph for a single fully connected or dense neural network.
    :param x: input tensor for the network with shape as defined in params.input_shape.
    :param params: ParameterBatch object containing all hyperparameters.
    :return: output tensor for the network with shape batch_size x 1.
    """
    input_layer = tf.reshape(x, [-1, params.input_shape[0], params.input_shape[1], params.input_shape[2]])
    flattened = tf.layers.flatten(input_layer, name="flatten_image")
    input_tensor = flattened

    def dense_layer(d_input, index):
        """
        Instanitates a new dense layer for the graph. This external function is neccesary for defining layers in
        a loop otherwise the graph is not built correctly.
        :param d_input: input tensor for the layer
        :param index: index of the layer to be created
        :return: output tensor of the layer
        """
        dense = tf.layers.dense(inputs=d_input,
                                units=int(params.n_dense_nodes / (index*params.dense_size_convergence + 1)),
                                activation=tf.nn.relu, name="dense_" + str(index))
        return dense

    for i in range(params.n_dense_layers):
        input_tensor = dense_layer(input_tensor, i)
    dense_no_activation = tf.layers.dense(inputs=input_tensor, units=20, activation=None, name="continuation_layer")
    out = tf.layers.dense(inputs=dense_no_activation, units=1, activation=None, name="output_node")
    return out


def spawn_network(params):
    """
    creates a single network instance.
    :param params: ParameterBatch object containing all hyperparameters.
    :return: either the best eval_loss achieved or the test_loss depending on params.testing.
    """

    try:

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu_id)

        def evaluation(ep):
            sess.run(eval_init_op)
            eval_batches = 0
            eval_loss = 0
            while True:
                try:
                    val_lss = sess.run(loss)
                    eval_batches += 1
                    eval_loss += val_lss
                except tf.errors.OutOfRangeError:
                    break
            eval_loss = eval_loss / eval_batches
            evaluation_logger.log_scalar("evaluation_loss", eval_loss, ep)
            return eval_loss

        def testing():
            sess.run(test_init_op)
            test_batches = 0
            test_loss = 0
            while True:
                try:
                    val_lss = sess.run(loss)
                    test_batches += 1
                    test_loss += val_lss
                except tf.errors.OutOfRangeError:
                    break
            test_loss = test_loss / test_batches
            return test_loss

        tf.reset_default_graph()
        tf.set_random_seed(1234)

        train_data = create_dataset(params.batch_size, params.train_csv_file)
        eval_data = create_dataset(params.batch_size, params.eval_csv_file)
        test_data = create_dataset(params.batch_size, params.test_csv_file)

        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        next_features, next_labels = iterator.get_next()
        train_init_op = iterator.make_initializer(train_data)
        eval_init_op = iterator.make_initializer(eval_data)
        test_init_op = iterator.make_initializer(test_data)

        if params.use_conv_net:
            prediction = cnn(next_features, params)
        else:
            prediction = fnn(next_features, params)

        element_loss = tf.losses.absolute_difference(
            labels=tf.expand_dims(next_labels, -1),
            predictions=prediction,
            reduction=tf.losses.Reduction.NONE)

        loss = tf.reduce_mean(element_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(loss)

        logdir = generate_logdir(params)

        training_logger = Logger(logdir + "/train")
        evaluation_logger = Logger(logdir + "/eval")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            overall_batches = 0
            eval_losses = []
            for epoch in range(params.n_max_epochs):
                epoch_loss = 0
                sess.run(train_init_op)
                while True:
                    try:
                        lss, _ = sess.run([loss, optimizer])
                        epoch_loss += lss
                        overall_batches += 1
                        training_logger.log_scalar("training_loss_batch", lss, overall_batches)
                    except tf.errors.OutOfRangeError:
                        break
                if epoch % 3 == 2:
                    params.logger.put("Evaluating performance in epoch " + str(epoch))
                    current_loss = float(evaluation(epoch))
                    eval_losses.append(current_loss)
                    better_results = [a for a in eval_losses if (a < current_loss) is True]
                    if len(better_results) >= 3:
                        params.logger.put("Evaluation Error has been increasing.")
                        break
            best_eval_loss = min(eval_losses)
            params.logger.put("Finished Training")

            if not params.testing:
                return best_eval_loss
            else:
                return testing()

    except tf.errors.ResourceExhaustedError:
        params.logger.put("Child on "+str(params.gpu_id)+" encountered OOM.")
        tf.reset_default_graph()
        return 2.0


def main():
    """
    This allows to run a standalone version of the network using the default parameters defined in _utils/ID_utils.py.
    :return:
    """
    gpu_id = check_available_gpus(max_n_gpus=1)[0]
    params = ParameterBatch(gpu_id=gpu_id, testing=False)
    command_str = "(rm -r " + params.logdir + ")"
    subprocess.run(command_str, shell=True)
    spawn_network(params)


if __name__ == "__main__":
    main()


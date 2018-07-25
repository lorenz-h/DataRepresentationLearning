import tensorflow as tf
import os

from _utils.ID_utils import check_available_gpus, get_input_shape, Colors, str2bool, Logger, get_convolutions
from ID_Input_Pipeline import create_dataset


def cnn(x, args):
    input_layer = tf.reshape(x, [-1, args.input_shape[0], args.input_shape[1], args.input_shape[2]])
    input_arr = input_layer
    for conv in args.convolutions:
        cnv = tf.layers.conv2d(
            inputs=input_arr,
            filters=conv[0],
            kernel_size=[conv[1], conv[2]],
            padding="same",
            activation=tf.nn.relu)
        input_arr = tf.layers.max_pooling2d(inputs=cnv, pool_size=[2, 2], strides=2)
    pool2_flat = tf.layers.flatten(input_arr)
    dense = tf.layers.dense(inputs=pool2_flat, units=args.n_dense_nodes, activation=None)
    dense_2 = tf.layers.dense(inputs=dense, units=int(args.n_dense_nodes/2), activation=None)
    out = tf.layers.dense(inputs=dense_2, units=1, activation=None)
    return out


def spawn_network(args):
    tf.reset_default_graph()
    train_data = create_dataset(args.batch_size, "_data/hetzell_training_data.csv")
    eval_data = create_dataset(args.batch_size, "_data/hetzell_evaluation_data.csv")
    test_data = create_dataset(args.batch_size, "_data/hetzell_testing_data.csv")

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)
    test_init_op = iterator.make_initializer(test_data)

    prediction = cnn(next_features, args)

    element_loss = tf.losses.absolute_difference(
        labels=tf.expand_dims(next_labels, -1),
        predictions=prediction,
        reduction=tf.losses.Reduction.NONE)

    loss = tf.reduce_mean(element_loss)

    accuracy = tf.reduce_mean(tf.cast(tf.less(element_loss, 0.63), dtype=tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    batch_avg_labels = tf.reduce_mean(tf.abs(next_labels))
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        def test_performance():
            sess.run(test_init_op)
            batches = 0
            test_loss = 0
            while True:
                try:
                    lss, acc = sess.run([loss, accuracy])
                    test_loss += lss
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            assert batches is not 0, "NO TEST DATA PASSED"
            test_loss /= batches
            return test_loss

        def evaluate_performance():
            sess.run(eval_init_op)
            batches = 0
            ev_loss = 0
            while True:
                try:
                    lss, acc = sess.run([loss, accuracy])
                    ev_loss = ev_loss + lss
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            assert batches is not 0, "NO EVALUATION DATA PASSED"
            ev_loss /= batches
            return ev_loss

        def training(lg_dir):
            training_logger = Logger(lg_dir+"/train")
            evaluation_logger = Logger(lg_dir + "/eval")
            eval_losses = []
            epoch = 0
            while epoch < args.n_max_epochs:
                epoch_loss = 0
                batches = 0
                sess.run(training_init_op)
                while True:
                    try:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                        epoch_loss = epoch_loss + lss
                        training_logger.log_scalar("epoch_training_loss" + str(args.gpu_id), epoch_loss, epoch)
                        batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                epoch_loss /= batches
                print("Epoch", epoch, "on GPU", args.gpu_id, "- Training loss was:", epoch_loss)
                if epoch % 5 == 1:
                    eval_loss = evaluate_performance()
                    evaluation_logger.log_scalar("evaluation_loss", eval_loss, epoch)
                    eval_losses.append(eval_loss)
                    print("Epoch", epoch, "on GPU", args.gpu_id, "- Evaluation loss was:", eval_loss)
                    print(len(list((lss < eval_loss for lss in eval_losses))))
                    if len(list((lss < eval_loss for lss in eval_losses))) > 5:
                        print("evaluation error has been increasing again.")
                        break
                epoch += 1
            smallest_eval_loss = sorted(eval_losses)[0]
            return smallest_eval_loss

        experiment_logdir = "gpu" + str(args.batch_size) + "_" + str(args.learning_rate) + "_" + str(args.gpu_id)
        logdir = os.path.join("_logs", experiment_logdir)
        if args.training:
            return training(logdir)
        else:
            training(logdir)
            return test_performance()


def setup_thread_environment(args):

    if args.automatic_gpu_placement:
        args.gpu_id = check_available_gpus()
        print(Colors.OKGREEN, "GPU", args.gpu_id, "is free and will be used.", Colors.ENDC)
    else:
        print(Colors.OKGREEN, "GPU", args.gpu_id, "has been parsed and will be used.", Colors.ENDC)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    acc = 0
    label_size = 0
    for it in range(0, args.n_runs):
        lss = spawn_network(args)
        acc += lss
        tf.reset_default_graph()  # this clears the graph
    acc = acc / args.n_runs
    return acc


def main():
    setup_thread_environment()


if __name__ == "__main__":
    main()



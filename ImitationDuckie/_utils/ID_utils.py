import subprocess
import argparse
import os
from os import listdir
import multiprocessing as mp
import csv
import time
from pathlib import Path

import tensorflow as tf
import skopt
import numpy as np
import logging


def check_available_gpus(max_n_gpus, logger):
    """
    This function runs nvidia-smi --id=gpu and captures the output. If it finds that there are no jobs running on a
    gpu it returns the gpus id.
    :return: the id of the first gpu found to be available
    """
    logger.put("searching for gpus...")
    system_gpus = 8
    gpus = []
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            gpus.append(gpu)
        if len(gpus) == max_n_gpus:
            break
    assert len(gpus) is not 0, "All GPUs are currently busy."
    logger.put("Using GPUs: " + str(gpus))
    return gpus


def str2bool(v):
    """
    This allows convenient parsing of bools.
    :param v: the string parsed through argparse
    :return: a bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Logger(object):
    """
    this allows convenient logging of python variables to tensorboard
    """
    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self.writer.add_graph(tf.get_default_graph())

    def log_scalar(self, tag, value, step):
        """
        Log a scalar variable
        :param tag: name of the scalar
        :param value: the scalar value
        :param step: the index at which the scalar value should be saved
        :return: ---
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_image(self, tag, images, step, sess):
        """
        :param tag: name of the images
        :param images: a stack of images of shape (n_images, x, y, channels)
        :param step: the index at which the images should be saved
        :param sess: the session in which log_image is called
        :return: ---
        """
        img_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        summary_op = tf.summary.image(name=tag, tensor=img_tensor, max_outputs=1)
        img_summary = sess.run(summary_op)
        self.writer.add_summary(img_summary, step)


def clear_logdir(logdir):
    """
    :param logdir: the relative path to the logdir to be deleted
    :return: ---
    """
    command_str = "(rm -r " + logdir + ")"
    subprocess.run(command_str, shell=True)
    command_str = "(mkdir " + logdir + ")"
    subprocess.run(command_str, shell=True)

    print("Empty Directory", logdir, "is available.")


def calc_average_label_size(dataset_dir):
    """
    :param dataset_dir: the directory for which to compute the average size of the labels
    :return: the average size of the labels in dataset_dir
    """
    labels = []
    for subdir in ("Training", "Testing"):
        path = dataset_dir + "/" + subdir + "/"
        for file in os.listdir(path):
            if file.endswith(".txt"):
                f = open(path + file, 'r')
                try:
                    label = f.read()
                    labels.append(float(label))
                finally:
                    f.close()
    arr_labels = np.array(labels)
    arr_labels = (arr_labels+1)*10
    std_label = np.std(arr_labels)
    print(std_label)
    avg_label = np.mean(np.absolute(arr_labels))
    return avg_label


def get_convolutions(n_convs, conv_size_divergion):
    convolutions = []
    for i in range(1, n_convs+1):
        convolutions.append((int(i*conv_size_divergion*16), 3, 3))
    return convolutions


def list_files(directory, extension):
    """
    lists all files in directory with extension extension.
    :param directory: the directory to search.
    :param extension: the extension of the files to list without the point
    :return:
    """
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def grayscale(array):
    """
    converts an rgb image to grayscale
    :param array: the image to convert
    :return: the 2D grayscale image
    """
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


class FauxLogger:
    def put(self, msg):
        print(msg)


class MessageLogger:
    """
    This is neccesary, because by default python logging is not process save.
    """
    def __init__(self, logdir):
        filename = logdir / 'optimizer_debug.log'
        logging.basicConfig(filename=filename, level=logging.DEBUG)

    def put(self, msg):
        logging.debug(msg)


class ParameterBatch:
    """
    Class of Object containing all hyperparameters needed to run the network.
    """
    def __init__(self,
                 learning_rate=0.0005,
                 input_shape=[80, 60, 1],
                 batch_size=24,
                 convolutions=[(64, 7, 7), (128, 5, 5)],
                 conv_size_divergence=1.0,
                 gpu_id=2,
                 n_dense_layers=4,
                 n_dense_nodes=128,
                 n_max_epochs=90,
                 dense_size_convergence=1.3,
                 testing=False,
                 use_conv_net=False,
                 n_runs=1,
                 train_csv_file=Path("_data/hetzell_shearlet_training_data.csv"),
                 eval_csv_file=Path("_data/hetzell_shearlet_evaluation_data.csv"),
                 test_csv_file=Path("_data/hetzell_shearlet_testing_data.csv"),
                 logdir=Path("_logs")
                 ):
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.convolutions = convolutions
        self.gpu_id = gpu_id
        self.n_dense_nodes = n_dense_nodes
        self.n_max_epochs = n_max_epochs
        self.n_runs = n_runs
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.eval_csv_file = eval_csv_file
        self.logger = FauxLogger()
        self.testing = testing
        self.use_conv_net = use_conv_net
        self.n_dense_layers = n_dense_layers
        self.dense_size_convergence = dense_size_convergence
        self.conv_size_divergence = conv_size_divergence
        self.lodir = logdir


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("PLease answer using")


class LoggableOptimizer(skopt.Optimizer):
    """
    This class inherits from the skopt Optimizer class. All it does is add logging functionality.
    """
    def __init__(self, dimensions, random_state, logdir, csv_file='optimizer_results.csv', logging_bool=True):
        skopt.Optimizer.__init__(self, dimensions=dimensions, random_state=random_state)
        self._lock = mp.Lock()
        self.logging = logging_bool
        self.log_file = logdir / csv_file

    def log_state(self):
        with self._lock:
            results = list(zip(self.yi, self.Xi))
            with open(self.log_file, 'w') as file:
                writer = csv.writer(file, delimiter=';')
                for row in results:
                    to_print = []
                    result = row[0]
                    setup = row[1]
                    to_print.append(result)
                    for param in setup:
                        to_print.append(param)
                    writer.writerow(to_print)


def map_point_to_param_batch(vals, logger, testing):

    params = ParameterBatch()
    params.gpu_id = vals[1]
    params.logger = logger
    params.testing = testing

    if params.use_conv_net:
        params.learning_rate = vals[0][0]
        params.convolutions = get_convolutions(vals[0][1], vals[0][3])
        params.n_dense_nodes = vals[0][2]
    else:
        params.learning_rate = vals[0][0]
        params.n_dense_nodes = vals[0][1]
        params.dense_size_convergence = vals[0][2]
        params.n_dense_layers = vals[0][3]
    return params


def log_default_params(logger):
    default_params = ParameterBatch()
    attrs = vars(default_params)
    print(attrs)
    logger.put("#############################################")
    logger.put("HYPERPARAMETERS:")
    for item in attrs.items():
        logger.put(str(item[0]) + " : " + str(item[1]))
    logger.put("#############################################")


def limit_size(list_obj, max_size):
    if len(list_obj) < max_size:
        return list_obj
    else:
        for i in range(len(list_obj)-max_size):
            list_obj.pop(0)
        return list_obj


def generate_logdir(args):
    if args.use_conv_net:
        experiment_logdir = f"gpu{args.gpu_id}_{args.learning_rate}_{args.n_dense_nodes}_" \
                            f"{args.n_dense_nodes}_{len(args.convolutions)}_{args.testing}"

    else:
        experiment_logdir = f"gpu{args.gpu_id}_{args.learning_rate}_{args.n_dense_nodes}_" \
                            f"{args.dense_size_convergence}_{args.n_dense_layers}_{args.testing}"

    exp_logdir = Path(args.logdir / experiment_logdir)
    return exp_logdir


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def stop(self):
        stop_time = time.time()
        return (stop_time-self.start_time)/3600

    def reset(self):
        self.start_time = time.time()

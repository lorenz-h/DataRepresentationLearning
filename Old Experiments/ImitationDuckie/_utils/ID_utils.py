import subprocess
import argparse
import os
from os import listdir
import multiprocessing as mp
import csv
import sys

import tensorflow as tf
import skopt
import numpy as np


def check_available_gpus():
    """
    This function runs nvidia-smi --id=gpu and captures the output. If it finds that there are no jobs running on a
    gpu it returns the gpus id.
    :return: the id of the first gpu found to be available
    """
    system_gpus = 8
    gpus = []
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            gpus.append(gpu)
    assert len(gpus) is not 0, "All GPUs are currently busy."
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
    command_str = "rm -r " + logdir
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
                 n_max_epochs=50,
                 dense_size_convergence=1.3,
                 testing=False,
                 n_runs=1,
                 train_csv_file="_data/hetzell_dct_training_data.csv",
                 eval_csv_file="_data/hetzell_dct_evaluation_data.csv",
                 test_csv_file="_data/hetzell_dct_testing_data.csv"
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
        self.logger = None
        self.testing = testing
        self.use_conv_net = True
        self.n_dense_layers = n_dense_layers
        self.dense_size_convergence = dense_size_convergence
        self.conv_size_divergence = conv_size_divergence


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")


class LoggableOptimizer(skopt.Optimizer):
    """
    This class inherits from the skopt Optimizer class. All it does is add logging functionality.
    """
    def __init__(self, dimensions, random_state, csv_file='_logs/optimizer_results.csv', logging=True):
        skopt.Optimizer.__init__(self, dimensions=dimensions, random_state=random_state)
        self._lock = mp.Lock()
        self.logging = logging
        self.log_file = csv_file

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


def c_print(*string, color=None):
    print_string = ""
    for st in string:
        print_string += " " + str(st)
    end_color = '\033[0m'
    colors = {'blue': '\033[94m', 'green': '\033[32m', 'red': '\033[31m'}
    assert color in colors, print("Argument", color,
                                  "not allowed. Allowed codes are:", [key for key, _ in colors.items()])
    print(colors.get(color), print_string, end_color)


def map_val_to_param_batch(vals):
    """
    Maps the values given by an Optimizer into a ParameterBatch object.
    :param vals: list of values from Optimizer
    :return: ParameterBatch object
    """
    params = ParameterBatch()

    if params.use_conv_net:
        params.learning_rate = vals[0]
        params.convolutions = get_convolutions(vals[1], vals[3])
        params.n_dense_nodes = vals[2]
    else:
        params.learning_rate = vals[0]
        params.n_dense_nodes = vals[1]
        params.dense_size_convergence = vals[2]
        params.n_dense_layers = vals[3]
    return params


def map_point_to_param_batch(vals):
    """
    Maps the values given by an Optimizer into a ParameterBatch object.
    :param vals: list of values from Optimizer
    :return: ParameterBatch object
    """
    params = ParameterBatch()

    if params.use_conv_net:
        params.learning_rate = vals[0]
        params.convolutions = get_convolutions(vals[1], vals[3])
        params.n_dense_nodes = vals[2]
        params.gpu_id = vals[4]
    else:
        params.learning_rate = vals[0]
        params.n_dense_nodes = vals[1]
        params.dense_size_convergence = vals[2]
        params.n_dense_layers = vals[3]
        params.gpu_id = vals[4]
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


# ########################LEGACY_CODE###################################################
"""
This code is not used in the final versions and only preserved to ensure compatibility with older versions
"""


class Colors:
    """
    used to color terminal output for better legibility.
    """
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class ThreadSaveCounter:
    def __init__(self, initial=0, maxvalue=None):
        """Initialize a new thread save counter to given initial value (default 0)."""
        self.value = initial
        self._lock = mp.Lock()
        self.maxValue = maxvalue

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

    def reached_upper_limit(self):
        if self.maxValue is not None:
            if self.value > self.maxValue:
                return True
            else:
                return False
        else:
            return False

import subprocess
import argparse
import tensorflow as tf
import os
import numpy as np
from os import listdir
import multiprocessing as mp


class Colors:
    """
    used to color terminal output for better legibility.
    """
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


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


def get_convolutions(n_convs):
    """
    :param n_convs: the number of kernels to produce
    :return: a list of kernels of shape (conv_depth, kernel_size_x, kernel_size_y)
    """
    convolutions = []
    for i in range(1, n_convs+1):
        convolutions.append((i*32, 5, 5))
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
                 input_shape=[480, 640, 1],
                 batch_size=16,
                 convolutions=[(64, 7, 7), (128, 5, 5)],
                 gpu_id=2,
                 n_dense_nodes=128,
                 n_max_epochs=30,
                 n_runs=1,
                 training=True,
                 train_csv_file="_data/hetzell_shearlet_training_data.csv",
                 eval_csv_file="_data/hetzell_shearlet_evaluation_data.csv",
                 test_csv_file="_data/hetzell_shearlet_testing_data.csv"
                 ):
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.convolutions = convolutions
        self.gpu_id = gpu_id
        self.n_dense_nodes = n_dense_nodes
        self.n_max_epochs = n_max_epochs
        self.n_runs = n_runs
        self.training = training
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.eval_csv_file = eval_csv_file


class ThreadSaveCounter:
    def __init__(self, initial=0, maxvalue=None):
        """Initialize a new atomic counter to given initial value (default 0)."""
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

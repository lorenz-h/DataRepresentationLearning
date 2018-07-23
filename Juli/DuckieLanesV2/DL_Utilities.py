import subprocess
from scipy.misc import imread
import argparse
import tensorflow as tf
import os
import numpy as np


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
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            return gpu
    assert False, "All GPUs are currently busy."


def get_input_shape(folder):
    """
    :param folder: The folder to open the first Training sample from
    :return: the shape of that first training sample
    """
    file_string = folder + "/Training/sample0.png"
    image = imread(file_string)
    # plt.imshow(image)
    # plt.show()
    imshape = image.shape
    assert len(imshape) > 1
    if len(imshape) == 2:
        imshape = imshape + (1,)
    return imshape


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


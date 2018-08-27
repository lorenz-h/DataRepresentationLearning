#!/usr/bin/env python
"""
This will launch a process_manager that launches a logger process that listens to a queue of logging events from
all other processes. Then some workers will be spawned that grab hyperparameter batches from a queue that gets populated
by the optimizer process. The process_manager will wait to recieve the results from the optimizer process. Once it has
recieved these results it will test the best setups.
"""

import tensorflow as tf
from skopt.space import Real, Integer

import time
import multiprocessing as mp
from queue import Empty
import subprocess
import logging
import csv

from ID_CNN_V01 import setup_process_environment
from _utils.ID_utils import check_available_gpus, LoggableOptimizer, map_val_to_param_batch, \
    ParameterBatch, log_default_params

__author__ = "Lorenz Hetzel"
__copyright__ = "Copyright 2018, Lorenz Hetzel"
__credits__ = ["Lorenz Hetzel", "Julian Zilly"]
__license__ = "GNU GPLv3"
__version__ = "1.0"
__maintainer__ = "Lorenz Hetzel"
__email__ = "hetzell@student.ethz.ch"

default_params = ParameterBatch()

if default_params.use_conv_net:
    dim_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
    dim_n_convolutions = Integer(low=1, high=3, name='n_convolutions')
    dim_conv_dense_nodes = Integer(low=128, high=1028, name='n_dense_nodes')
    dim_conv_depth_divergence = Real(low=0.8, high=2.0, name='conv_size_divergence')

    optimizer_dimensions = [dim_learning_rate, dim_n_convolutions, dim_conv_dense_nodes, dim_conv_depth_divergence]
else:
    dim_dense_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
    dim_dense_nodes = Integer(low=312, high=1028, name='n_dense_nodes')
    dim_dense_size_convergence = Real(low=0.7, high=2.0, name='dense_size_convergence')
    dim_dense_n_layers = Integer(low=3, high=6, name='n_layers')

    optimizer_dimensions = [dim_dense_learning_rate, dim_dense_nodes, dim_dense_size_convergence, dim_dense_n_layers]

max_n_points = 40
lock = mp.Lock()


def logging_process(logger, testing):
    """
    :param logger: The Queue from which to grab the log events.
    :param testing: Bool indicating wether the logger will be used for testing or for training.
    :return: ---
    """
    if testing:
        filename = '_logs/optimizer_test_debug.log'
    else:
        filename = '_logs/optimizer_debug.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    sec = 0
    while True:
        try:
            msg = logger.get(block=False)
            msg = str(msg)
            logging.info(msg)
        except Empty:
            time.sleep(1)
            sec += 1
            if sec % 120 == 0:
                logging.info("Logger is still alive after " + str(sec) + " seconds.")


def worker_process(gpu, logger, queue, results_queue, training=True):
    time.sleep(2)
    logger.put("Started Child" + str(gpu))
    while True:
        try:
            point = queue.get(block=True)
            params = map_val_to_param_batch(point)
            params.gpu_id = gpu
            params.logger = logger
            params.training = training
            lss = setup_process_environment(params)
            with lock:
                results_queue.put(["info", point, lss])
        except tf.errors.ResourceExhaustedError:
            logger.put("Child" + str(gpu) + "encountered OOM")
            with lock:
                results_queue.put(["info", point, 2.0])


def optimizer_process(logger, reserved_gpus, queue, results_queue, child_node):
    """
    :param logger:  The Queue to write logging events to.
    :param reserved_gpus: The GPUs available to the process
    :param queue: The queue to put the setups provided by the LoggableOptimizer into
    :param results_queue: The Queue from which to get the results of the workers to tell to the LoggableOptimizer
    :param child_node: The Pipeline to the process_manager, this will be used to deliver final results of the optimizer
    :return: ---
    """
    optimizer = LoggableOptimizer(dimensions=optimizer_dimensions, random_state=1)
    n_points_solved = 0
    n_points_enqueued = len(reserved_gpus)+2
    for i in range(len(reserved_gpus)+2):
        queue.put(optimizer.ask())
    while True:
        try:
            msg = results_queue.get(block=False)
            if msg[0] == 'info':
                optimizer.tell(msg[1], msg[2])
                n_points_solved += 1
                logger.put(str(n_points_solved) + " out of " + str(max_n_points) + " Points have been solved")
                optimizer.log_state()
                if n_points_enqueued < max_n_points:
                    queue.put(optimizer.ask())
        except Empty:
            time.sleep(10)
        if n_points_solved >= max_n_points:
            break
    sorted_eval_results = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    logger.put(str(sorted_eval_results))
    child_node.send(sorted_eval_results)
    child_node.close()


def testing_best_setups(eval_results, reserved_gpus):
    """
    :param eval_results: Tbe Results of the Optimizer
    :param reserved_gpus: The GPUs available
    :return: ---
    """
    n_points_to_test = 3

    test_points_queue = mp.Queue()
    test_results_queue = mp.Queue()
    test_logger = mp.Queue()

    test_log_daemon = mp.Process(target=logging_process, args=(test_logger, True,), daemon=True, name="TestLogger")
    test_log_daemon.start()
    test_logger.put('Started Testing Logging Daemon.')
    test_logger.put(eval_results)

    for i in range(n_points_to_test):
        test_points_queue.put(eval_results[i][1])

    test_workers = []
    for i in range(len(reserved_gpus)):
        p = mp.Process(target=worker_process, args=(reserved_gpus[i], test_logger,
                                                    test_points_queue, test_results_queue, True,))
        p.start()
        test_workers.append(p)

    n_points_tested = 0
    while n_points_tested < n_points_to_test:
        try:
            msg = test_results_queue.get(block=False)
            if msg[0] == 'info':
                n_points_tested += 1

        except Empty:
            time.sleep(7)

    for p in test_workers:
        p.terminate()
        p.join()
    test_log_daemon.terminate()
    test_log_daemon.join()

    with open("_logs/test_results.csv", 'w') as file:
        writer = csv.writer(file, delimiter=';')
        for row in eval_results:
            to_print = []
            result = row[0]
            setup = row[1]
            to_print.append(result)
            for param in setup:
                to_print.append(param)
            writer.writerow(to_print)


def process_manager():
    subprocess.run("(rm -r _logs)", shell=True)
    subprocess.run("(mkdir _logs)", shell=True)

    reserved_gpus = check_available_gpus()
    reserved_gpus.pop(0)
    logger = mp.Queue()
    log_daemon = mp.Process(target=logging_process, args=(logger, False, ), daemon=True, name="Logger")
    log_daemon.start()
    logger.put('Started Logging Daemon.')

    log_default_params(logger)

    queue = mp.Queue()
    results_queue = mp.Queue()
    child_node, parent_node = mp.Pipe()

    workers = []
    for i in range(len(reserved_gpus)):
        p = mp.Process(target=worker_process, args=(reserved_gpus[i], logger, queue, results_queue, True, ))
        p.start()
        workers.append(p)

    optimizer_daemon = mp.Process(target=optimizer_process, args=(logger, reserved_gpus, queue,
                                                                  results_queue, child_node), daemon=True, name="Logger")
    optimizer_daemon.start()
    logger.put('Started Optimizer Daemon.')
    try:
        optimizer_results = parent_node.recv()  # wait to recieve results from optimizer
        for p in workers:
            p.terminate()
            p.join()

        testing_best_setups(optimizer_results, reserved_gpus)
        logger.put("Testing Ran Sucessfully Exiting...")
        time.sleep(2)
        log_daemon.terminate()
        log_daemon.join()
    except EOFError:
        logger.put("Optimizer crashed...")


if __name__ == "__main__":
    process_manager()

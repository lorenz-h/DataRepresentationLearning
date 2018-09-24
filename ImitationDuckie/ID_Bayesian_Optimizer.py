#!/usr/bin/env python
"""
This file contains all the definitons for an Bayesian optimization cycle, mainly in start_optimization. Wether a
ConvNet or a DenseNet is used is decided by boolean default_params.use_conv_net. The datasets that will be used are
also defined in default_params.
"""
from skopt.space import Real, Integer

import multiprocessing as mp
import csv
import time

from ID_Network_Graph import spawn_network
from _utils.ID_utils import check_available_gpus, LoggableOptimizer, map_point_to_param_batch,\
    ParameterBatch, log_default_params, MessageLogger, Timer, clear_logdir

#####################################################################################################################

max_n_points = 60  # the number of points that will be evaluated in hyperparameter space
max_n_gpus = 6  # the maximum number of gpus to use on rudolf.

default_params = ParameterBatch()  # load the default hyperparameters as defined in _utils.ID_utils.py


#  DEFINE DIMENSIONS AND CONSTRAINTS FOR THE BAYESIAN OPTIMIZATION
if default_params.use_conv_net:
    dim_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
    dim_n_convolutions = Integer(low=1, high=4, name='n_convolutions')
    dim_conv_dense_nodes = Integer(low=128, high=1500, name='n_dense_nodes')
    dim_conv_depth_divergence = Real(low=0.8, high=2.0, name='conv_size_divergence')

    optimizer_dimensions = [dim_learning_rate, dim_n_convolutions, dim_conv_dense_nodes, dim_conv_depth_divergence]
else:
    dim_dense_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
    dim_dense_nodes = Integer(low=312, high=1500, name='n_dense_nodes')
    dim_dense_size_convergence = Real(low=0.7, high=2.0, name='dense_size_convergence')
    dim_dense_n_layers = Integer(low=3, high=6, name='n_layers')

    optimizer_dimensions = [dim_dense_learning_rate, dim_dense_nodes, dim_dense_size_convergence, dim_dense_n_layers]

#####################################################################################################################


def start_optimization(reserved_gpus, logger):
    """
    runs an optimization on networks created by spawn_network. len(reserved_gpus) points are evaluated in parallel,
    unless the limit max_n_points is about to be reached. The results of the optimization are saved to a csv file in
    default_params.logdir. The ten best results are retrained and subsequently tested on the testing dataset and these
    results are also saved to a csv file in default_params.logdir.
    :param reserved_gpus: the gpus which are available for the optimizer to use
    :param logger: the logger the process should write to.
    :return: ---
    """
    timer = Timer()
    optimizer = LoggableOptimizer(dimensions=optimizer_dimensions, random_state=1, logdir=default_params.logdir)
    n_points_solved = 0

    while n_points_solved < max_n_points:
        batch_size = min(len(reserved_gpus), max_n_points - n_points_solved)
        logger.put("BATCHSIZE:"+str(batch_size))
        points = optimizer.ask(n_points=batch_size)
        points_with_gpus = list(zip(points, reserved_gpus[0:batch_size]))
        points_with_gpus = [map_point_to_param_batch(point, logger, testing=False) for point in points_with_gpus]

        p = mp.Pool(batch_size)
        lss = p.map(spawn_network, points_with_gpus)
        p.close()
        optimizer.tell(points, lss)
        optimizer.log_state()
        time.sleep(5)
        n_points_solved += batch_size
        logger.put(str(n_points_solved)+" Points have been solved")
    logger.put("Optimization ended after " + str(timer.stop()) + " hours")
    timer.reset()

    logger.put("Started Testing.")

    n_points_to_test = 10
    sorted_eval_results = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    points_to_test = [result[1] for result in sorted_eval_results][0:n_points_to_test]

    test_results = []
    while len(points_to_test) is not 0:
        batch = []
        for gpu in reserved_gpus:
            if len(points_to_test) is not 0:
                batch.append([points_to_test.pop(0), gpu])
        batch_with_gpus = [map_point_to_param_batch(point, logger, testing=True) for point in batch]

        p = mp.Pool(len(batch_with_gpus))
        lss = p.map(spawn_network, batch_with_gpus)
        test_results = test_results + list(zip(lss, batch))
        p.close()
    logger.put(test_results)
    logger.put("Testing ended after " + str(timer.stop()) + " hours")
    csv_file = default_params.logdir+'/testing_results.csv'
    with open(csv_file, 'w') as file:
        writer = csv.writer(file, delimiter=';')
        for row in test_results:
            to_print = []
            result = row[0]
            setup = row[1][0]
            to_print.append(result)
            for param in setup:
                to_print.append(param)
            writer.writerow(to_print)
    logger.put("Wrote to csv file.")


def main():
    clear_logdir(default_params.logdir)  # clear the logdir

    logger = MessageLogger(logdir=default_params.logdir)  # this is neccesary because python logging is not process save

    reserved_gpus = check_available_gpus(max_n_gpus=max_n_gpus, logger=logger)

    log_default_params(logger)  # just write the default params used to the logger

    start_optimization(reserved_gpus, logger)


if __name__ == "__main__":
    main()

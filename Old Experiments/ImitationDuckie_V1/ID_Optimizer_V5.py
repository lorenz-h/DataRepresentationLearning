
import tensorflow as tf
from skopt.space import Real, Integer

import time
import multiprocessing as mp
from queue import Empty
import subprocess
import logging
import csv
from queue import Empty
from sklearn.externals.joblib import Parallel

from ID_CNN_V01 import setup_process_environment
from _utils.ID_utils import check_available_gpus, LoggableOptimizer, map_point_to_param_batch,\
    ParameterBatch, log_default_params, limit_size


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

max_n_points = 6
max_n_gpus = 4


def optimizer_process(reserved_gpus):
    optimizer = LoggableOptimizer(dimensions=optimizer_dimensions, random_state=1)
    n_points_solved = 0

    while n_points_solved < max_n_points:
        points = optimizer.ask(n_points=len(reserved_gpus))
        points_with_gpus = []
        i = 0
        for point in points:
            points_with_gpus.append(point + [reserved_gpus[i]])
            i += 1
        print(points_with_gpus)
        lss = Parallel()(setup_process_environment(map_point_to_param_batch(point)) for point in points_with_gpus)
        optimizer.tell(points, lss)
        optimizer.log()
        n_points_solved += len(reserved_gpus)
        logging.debug(str(n_points_solved)+" Points have been solved")
    print("OPTI ENDED REGULARLY")


def main():
    optimizer_process([6, 7])


if __name__ == "__main__":
    main()

import multiprocessing as mp
from skopt import Optimizer
from skopt.space import Real, Integer
import subprocess
import time
import pickle

from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, Colors, check_available_gpus

dim_learning_rate = Real(low=1e-7, high=3e-2, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=1, high=4, name='n_convolutions')
dim_dense_nodes = Integer(low=64, high=256, name='n_dense_nodes')


class ParameterBatch:
    """
    Object containing all hyperparameters needed to run the network.
    """
    def __init__(self,
                 learning_rate=0.001,
                 input_shape=[480, 640, 1],
                 batch_size=32,
                 convolutions=[(64, 5, 5), (128, 5, 5)],
                 gpu_id=2,
                 n_dense_nodes=64,
                 n_max_epochs=30,
                 n_runs=1,
                 training=True):
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.convolutions = convolutions
        self.gpu_id = gpu_id
        self.n_dense_nodes = n_dense_nodes
        self.n_max_epochs = n_max_epochs
        self.n_runs = n_runs
        self.training = training


def map_val_to_param_batch(vals, gpu_id):
    """
    Maps the values given by an Optimizer into a ParameterBatch object.
    :param vals: list of values from Optimizer
    :param gpu_id: the gpu_id passed to the ParameterBatch
    :return: ParameterBatch object
    """
    params = ParameterBatch(learning_rate=vals[0],
                            convolutions=get_convolutions(vals[1]),
                            n_dense_nodes=vals[2],
                            gpu_id=gpu_id)
    return params


def bayesian_optimize(n_calls=2):
    """
    Apply bayesian optimization to a network. Access global variable reserved_gpus, Ask Optimizer for one point for each GPU,
    train and evaluate at that point in parallellized threads, repeat n_calls times. Then train and test the best setup.
    :return: ---
    """
    start_time = time.time()
    p = mp.Pool(len(reserved_gpus))
    optimizer = Optimizer(dimensions=[dim_learning_rate, dim_n_convolutions, dim_dense_nodes],
                          random_state=1)
    for i in range(1, n_calls + 1):
        gpus = list(reserved_gpus)
        vals = optimizer.ask(n_points=len(reserved_gpus))
        points = []
        for point in vals:
            param_batch = map_val_to_param_batch(point, gpus.pop(0))
            points.append(param_batch)
        loss = p.map(setup_thread_environment, points)
        optimizer.tell(vals, loss)
        print("#" * 100)
        print(Colors.OKBLUE, "Optimization cylce", i, "done.", Colors.ENDC)
        print("#" * 100)
    print("Best setup found:")
    p.close()
    print(min(optimizer.yi))  # print the best objective found
    sorted_sets = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    print("BEST SET:", sorted_sets[0])
    gpus = list(reserved_gpus)
    test_args = map_val_to_param_batch(sorted_sets[0][1], gpus.pop(0))
    test_args.training = False
    avg_test_accuracy = setup_thread_environment(test_args)
    print("Test accuracy:", avg_test_accuracy)
    end_time = time.time()
    print("It took:", str(end_time - start_time), "seconds")
    pickle.dump(sorted_sets, open("_logs/optimizer_points.pkl", "wb"))
    try:
        file_path = "_logs/dl_optimizer_result.txt"
        label_file = open(file_path, "w")
        label_file.write("Best setup found:\n")
        label_file.write(str(sorted_sets[0]))
        label_file.write("\nTime to process: ")
        label_file.write(str(end_time - start_time))
        label_file.write("\nTest Accuracy: ")
        label_file.write(str(avg_test_accuracy))
    finally:
        label_file.close()


def main():
    """
    clears the logdir, finds the reserved_gpus and starts the bayesian optimization.
    :return: ---
    """
    global reserved_gpus
    command_str = "(rm -r _logs)"
    subprocess.run(command_str, shell=True)
    reserved_gpus = check_available_gpus()
    print("GPUs", reserved_gpus, "are available.")
    bayesian_optimize()


if __name__ == "__main__":
    main()

import multiprocessing as mp
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, Colors, check_available_gpus
from skopt import Optimizer
from skopt.space import Real, Integer
import subprocess

n_calls = 2

dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=1, high=4, name='n_convolutions')
dim_dense_nodes = Integer(low=32, high=128, name='n_dense_nodes')


class ParameterBatch:
    def __init__(self,
                 automatic_gpu_placement=False,
                 learning_rate=0.001,
                 input_shape=[480, 640, 3],
                 batch_size=32,
                 convolutions=[(32, 5, 5), (64, 5, 5)],
                 gpu_id=2,
                 n_dense_nodes=32,
                 n_max_epochs=2,
                 n_runs=1,
                 training=True):
        self.automatic_gpu_placement = automatic_gpu_placement
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
    params = ParameterBatch(learning_rate=vals[0],
                            convolutions=get_convolutions(vals[1]),
                            n_dense_nodes=vals[2],
                            gpu_id=gpu_id)
    return params


def bayesian_optimize():
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


def main():
    global reserved_gpus
    command_str = "(rm -r _logs)"
    subprocess.run(command_str, shell=True)
    reserved_gpus = check_available_gpus()
    print("GPUs", reserved_gpus, "are available.")
    bayesian_optimize()


if __name__ == "__main__":
    main()

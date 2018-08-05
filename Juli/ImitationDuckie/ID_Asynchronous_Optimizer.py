import time
from skopt import Optimizer
from skopt.space import Real, Integer
import pickle
import multiprocessing as mp
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, Colors, check_available_gpus, ParameterBatch, ThreadSaveCounter
import subprocess

dim_learning_rate = Real(low=1e-7, high=3e-2, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=1, high=3, name='n_convolutions')
dim_dense_nodes = Integer(low=128, high=200, name='n_dense_nodes')


queue = mp.Queue()


n_points = ThreadSaveCounter(maxvalue=15)

lock = mp.Lock()
optimizer = Optimizer(dimensions=[dim_learning_rate, dim_n_convolutions, dim_dense_nodes], random_state=1)


def map_val_to_param_batch(vals):
    """
    Maps the values given by an Optimizer into a ParameterBatch object.
    :param vals: list of values from Optimizer
    :param gpu_id: the gpu_id passed to the ParameterBatch
    :return: ParameterBatch object
    """
    params = ParameterBatch(learning_rate=vals[0],
                            convolutions=get_convolutions(vals[1]),
                            n_dense_nodes=vals[2])
    return params


def tt(gpu):
    """
    This is one thread. It will take a point from the queue, evaluate it and tell the optimizer the result.
    It will then ask the optimizer for a new point and append it to the queue. It will exit once a certain number of
    points has been evaluated.
    :param gpu: The GPU ID the thread should use
    :return: ---
    """
    print("Started Thread", gpu)
    while True:
        point = queue.get()
        params = map_val_to_param_batch(point)
        params.gpu_id = gpu
        lss = setup_thread_environment(params)
        with lock:
            optimizer.tell(point, lss)
            new_point = optimizer.ask()
            n_points.increment()
        queue.put(new_point)
        print(Colors.WARNING, gpu, ": Sucessfully Investigated", point, Colors.ENDC)
        if n_points.reached_upper_limit():
            break


def spawn_threads():
    """
    This will look for available gpus and then spawn a process tt for each of the free gpus. After the optimization
    is complete it will test the best setup found and save the results.
    :return: ---
    """
    command_str = "(rm -r _logs)"
    subprocess.run(command_str, shell=True)
    command_str = "(rm -r nohup.out)"
    subprocess.run(command_str, shell=True)
    start_time = time.time()
    reserved_gpus = check_available_gpus()
    print(Colors.OKGREEN, "GPUs", reserved_gpus, "are available.", Colors.ENDC)
    p = mp.Pool(len(reserved_gpus))
    for i in range(len(reserved_gpus)+1):
        queue.put(optimizer.ask())
    p.map(tt, reserved_gpus)
    p.close()
    print("All Threads finished.")
    sorted_sets = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    print("BEST SET:", sorted_sets[0])
    print("STARTING TESTING")
    test_args = map_val_to_param_batch(sorted_sets[0][1], reserved_gpus[0])
    test_args.training = False
    avg_test_accuracy = setup_thread_environment(test_args)
    print("Test accuracy:", avg_test_accuracy)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("It took", time_elapsed, "seconds.")
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
    spawn_threads()


if __name__ == "__main__":
    main()

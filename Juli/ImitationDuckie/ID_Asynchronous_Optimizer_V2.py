import time
from skopt.space import Real, Integer
import multiprocessing as mp
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, Colors, check_available_gpus, ParameterBatch, LoggableOptimizer
import subprocess

dim_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=1, high=3, name='n_convolutions')
dim_dense_nodes = Integer(low=128, high=200, name='n_dense_nodes')

max_n_points = 70
queue = mp.Queue()
lock = mp.Lock()


def map_val_to_param_batch(vals):
    """
    Maps the values given by an Optimizer into a ParameterBatch object.
    :param vals: list of values from Optimizer
    :return: ParameterBatch object
    """
    params = ParameterBatch(learning_rate=vals[0],
                            convolutions=get_convolutions(vals[1]),
                            n_dense_nodes=vals[2])
    return params


def child(gpu, child_node):
    print("Started Child", gpu)
    while True:
        if queue.empty():
            break
        point = queue.get()
        params = map_val_to_param_batch(point)
        params.gpu_id = gpu
        lss = setup_thread_environment(params)
        with lock:
            child_node.send([point, lss])
        print(Colors.WARNING, "GPU", gpu, ": Sucessfully Investigated", point, Colors.ENDC)
        time.sleep(1)  # so that the server_process has enough time to enqueue a new point
    print(Colors.WARNING, "Child", gpu, "exited.", Colors.ENDC)


def server_process():
    command_str = "(rm -r _logs)"
    subprocess.run(command_str, shell=True)
    optimizer = LoggableOptimizer(dimensions=[dim_learning_rate, dim_n_convolutions, dim_dense_nodes], random_state=1)
    parent_node, child_node = mp.Pipe()
    reserved_gpus = check_available_gpus()
    for i in range(len(reserved_gpus)+2):
        queue.put(optimizer.ask())
    processes = []
    for i in range(len(reserved_gpus)):
        p = mp.Process(target=child, args=(reserved_gpus[i], child_node))
        p.start()
        processes.append(p)
    i = len(reserved_gpus)

    while True:
        time.sleep(3)
        children_busy = False
        for p in processes:
            if p.is_alive():
                children_busy = True
        if not children_busy:
            break
        msg = parent_node.recv()  # waits to recieve a message from one of the children
        print(Colors.OKGREEN, "Server Recieved info about", msg[0], msg[1], Colors.ENDC)
        optimizer.tell(msg[0], msg[1])
        optimizer.log_state()
        i += 1
        if i < max_n_points:
            queue.put(optimizer.ask())
            print(Colors.OKGREEN, "Generated new point", Colors.ENDC)

    for p in processes:
        p.join()

    sorted_sets = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    print(Colors.OKBLUE, "Finished Optimization. Best Set:", sorted_sets[0], Colors.ENDC)
    print(Colors.OKBLUE, "Results have been stored. Commencing Testing...", Colors.ENDC)
    test_args = map_val_to_param_batch(sorted_sets[0][1])
    test_args.training = False
    test_args.gpu_id = reserved_gpus[0]
    test_args.n_runs = 2
    avg_test_accuracy = setup_thread_environment(test_args)
    with open("_logs/final_result.txt", 'w') as file:
        file.write("Testing Accuracy:"+"\n")
        file.write(str(avg_test_accuracy)+"\n")
        file.write("With Setup" + "\n")
        file.write(str(sorted_sets[0][1]))
    print(Colors.OKBLUE, "Finished Testing. Server is Shutting Down", Colors.ENDC)


def main():
    server_process()


if __name__ == "__main__":
    main()

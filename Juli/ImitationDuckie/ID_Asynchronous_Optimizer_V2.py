import time
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, c_print, check_available_gpus, ParameterBatch, LoggableOptimizer
import subprocess
import tensorflow as tf

dim_learning_rate = Real(low=1e-10, high=3e-1, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=3, high=5, name='n_convolutions')
dim_dense_nodes = Integer(low=300, high=512, name='n_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'tanh', 'None'])
max_n_points = 100
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
    """
    :param gpu: The GPU the child should use.
    :param child_node: the pipeline endpoint through which the child will send and recieve information to the server
    :return: ---
    """
    try:
        print("Started Child", gpu)
        while True:
            if queue.empty():
                break
            point = queue.get()
            params = map_val_to_param_batch(point)
            params.gpu_id = gpu
            lss = setup_thread_environment(params)
            child_node.send([point, lss])
            c_print("GPU", gpu, ": Sucessfully Investigated", point, color="red")
            time.sleep(1)  # so that the server_process has enough time to enqueue a new point
        c_print("Child", gpu, "exited.", color="red")
    except tf.errors.ResourceExhaustedError:
        child_node.send([point, 0.09])
        child_node.send([gpu, "crashed"])


def server_process():
    """
    Create a new Optimizer, Create Queue for Values to be investigated by Optimizer.
    Spawn subprocesses (children) listen for their results enqueue new points until max_n_points is reached.
    Exit the loop when all children have terminated.
    :return: ---
    """
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
        """
                children_busy = False
                for p in processes:
            if p.is_alive():
                children_busy = True
        if not children_busy:
            break
        """
        msg = parent_node.recv()  # waits to recieve a message from one of the children
        c_print(msg, color="blues")
        if msg[1] is "crashed":
            if not queue.empty():
                c_print("Launching new child on gpu", msg[0], color="blue")
                p = mp.Process(target=child, args=(msg[0], child_node))
                p.start()
                processes.append(p)

        c_print("Server Recieved info about", msg[0], msg[1], color='green')
        optimizer.tell(msg[0], msg[1])
        optimizer.log_state()
        i += 1
        if i < max_n_points:
            queue.put(optimizer.ask())
            c_print("Generated new Point", color='green')

    for p in processes:
        p.join()

    sorted_sets = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    c_print("Finished Optimization. Best Set:", sorted_sets[0], color="blue")
    c_print("Results have been stored. Commencing Testing...", color="blue")
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
    c_print("Finished Testing. Server is Shutting Down", color="blue")


def main():
    server_process()


if __name__ == "__main__":
    main()

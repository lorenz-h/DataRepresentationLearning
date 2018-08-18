import time
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
import queue as queue_thrd
from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import get_convolutions, c_print, check_available_gpus, ParameterBatch, LoggableOptimizer
import subprocess
import tensorflow as tf
import logging

dim_learning_rate = Real(low=1e-10, high=3e-1, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=0, high=5, name='n_convolutions')
dim_dense_nodes = Integer(low=128, high=650, name='n_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'tanh', 'None'])
max_n_points = 15
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
    time.sleep(2)
    logging.info("Started Child" + str(gpu))
    try:
        while True:
                point = queue.get(block=True, timeout=15)
                params = map_val_to_param_batch(point)
                params.gpu_id = gpu
                lss = setup_thread_environment(params)
                with lock:
                    child_node.send(["info", point, lss])
    except tf.errors.ResourceExhaustedError:
        c_print("RESOURCE WAS EXHAUSTET", color="red")
        logging.warning("Child" + str(gpu) + "encountered OOM")
        with lock:
            child_node.send(["info", point, 1])
    except queue_thrd.Empty:
        logging.debug("Child" + str(gpu) + "terminated, because Queue was empty.")

    child_node.send(["note", gpu, "exited"])


def server_process():
    subprocess.run("(rm -r _logs)", shell=True)
    subprocess.run("(mkdir _logs)", shell=True)

    logging.basicConfig(filename='_logs/optimizer_debug.log', level=logging.DEBUG)
    logging.debug('Started Logging.')
    optimizer = LoggableOptimizer(dimensions=[dim_learning_rate, dim_n_convolutions, dim_dense_nodes], random_state=1)

    reserved_gpus = check_available_gpus()
    for i in range(len(reserved_gpus) + 2):
        queue.put(optimizer.ask())

    parent_node, child_node = mp.Pipe()

    processes = []
    points_counter = 0
    while points_counter < len(reserved_gpus):
        p = mp.Process(target=child, args=(reserved_gpus[points_counter], child_node))
        p.start()
        processes.append(p)
        points_counter += 1
    logging.debug(str(points_counter)+"Points have been logged.")
    while True:
        time.sleep(7)
        if any([p.is_alive() for p in processes]):
            try:
                msg = parent_node.recv()  # waits to recieve a message from one of the children
            except EOFError:  # if all children have exited this error will be raised
                break
        else:
            break
        print(msg)
        if msg[0] == 'info':
            logging.debug("Server Recieved Info" + str(msg))
            optimizer.tell(msg[1], msg[2])
            optimizer.log_state()
            if points_counter < max_n_points:
                logging.info("Enqueued Point Number " + str(points_counter))
                points_counter += 1
                queue.put(optimizer.ask())
        if msg[0] == 'note':
            processes.pop(0)
    logging.info("Server finished regularly.")


def main():
    server_process()


if __name__ == "__main__":
    main()

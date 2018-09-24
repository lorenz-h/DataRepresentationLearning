import tensorflow as tf
from skopt.space import Real, Integer

import time
import multiprocessing as mp
from queue import Empty
import subprocess
import logging

from ID_CNN_V01 import setup_thread_environment
from _utils.ID_utils import check_available_gpus, LoggableOptimizer, map_val_to_param_batch


dim_learning_rate = Real(low=1e-9, high=3e-1, prior='log-uniform', name='learning_rate')
dim_n_convolutions = Integer(low=2, high=5, name='n_convolutions')
dim_dense_nodes = Integer(low=128, high=512, name='n_dense_nodes')
max_n_points = 60
queue = mp.Queue()
lock = mp.Lock()


def logging_process(logger):
    logging.basicConfig(filename='_logs/optimizer_debug.log', level=logging.DEBUG)
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


def child_process(gpu, child_node, logger):
    time.sleep(2)
    logger.put("Started Child" + str(gpu))
    while True:
        try:
                point = queue.get(block=True, timeout=15)
                params = map_val_to_param_batch(point)
                params.gpu_id = gpu
                params.logger = logger
                lss = setup_thread_environment(params)
                with lock:
                    child_node.send(["info", point, lss])
        except tf.errors.ResourceExhaustedError:
            logger.put("Child" + str(gpu) + "encountered OOM")
            with lock:
                child_node.send(["info", point, 2.0])
        except Empty:
            logger.put("Child" + str(gpu) + "terminated, because Queue was empty.")
    with lock:
        child_node.send(["note", gpu, "exited"])


def server_process():
    subprocess.run("(rm -r _logs)", shell=True)
    subprocess.run("(mkdir _logs)", shell=True)
    logger = mp.Queue()
    log_daemon = mp.Process(target=logging_process, args=(logger,),
                            daemon=True, name="Logger")
    log_daemon.start()
    logger.put('Started Logging.')
    optimizer = LoggableOptimizer(dimensions=[dim_learning_rate, dim_n_convolutions, dim_dense_nodes], random_state=1)

    reserved_gpus = check_available_gpus()
    reserved_gpus.pop(0)
    reserved_gpus.pop(0)
    for i in range(len(reserved_gpus) + 2):
        queue.put(optimizer.ask())

    parent_node, child_node = mp.Pipe()

    processes = []
    points_counter = 0
    while points_counter < len(reserved_gpus):
        p = mp.Process(target=child_process, args=(reserved_gpus[points_counter], child_node, logger))
        p.start()
        processes.append(p)
        points_counter += 1
    logger.put(str(points_counter)+"Points have been logged.")

    while True:  # this loop interacts with the children until all points have been evaluated.
        time.sleep(7)
        for p in processes:
            logger.put(str(p.name) + " is alive: " + str(p.is_alive()))
            if points_counter < max_n_points:
                if not p.is_alive():  # if a child is dead restart it and add point to queue.
                    queue.put(optimizer.ask())
                    points_counter += 1
                    logger.put("Enqueued Point Number " + str(points_counter))
                    p.start()
                    logger.put("Relaunched Child")
        if any([p.is_alive() for p in processes]):
            try:
                msg = parent_node.recv()  # waits to recieve a message from one of the children
            except EOFError:  # if all children have exited this error will be raised
                break
        else:
            break
        logger.put("Server Recieved Message" + str(msg))
        if msg[0] == 'info':
            logger.put(str(points_counter) + " out of " + str(max_n_points) +
                       " Points have been logged")
            optimizer.tell(msg[1], msg[2])
            optimizer.log_state()
            if points_counter < max_n_points:
                logger.put("Enqueued Point Number " + str(points_counter))
                points_counter += 1
                queue.put(optimizer.ask())
    logger.put("Server finished regularly.")
    time.sleep(2)  # wait for logger to log before shutting it down
    log_daemon.terminate()
    log_daemon.join()


def main():
    server_process()


if __name__ == "__main__":
    main()

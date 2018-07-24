from skopt import Optimizer
from skopt.space import Real, Integer
from multiprocessing import Pool
import subprocess
import time
import pickle

n_gpus = 5
# this specifies the number of gpus that should be used in paralell to evaluate the objective.
n_calls = 4
# n_calls specifies how many optimization steps the bayesian optimizer will take.
n_runs = 1
# n_runs specifies how often the network will be evaluated at one point to account for the randomly initialized weights

dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_n_epochs = Integer(low=2, high=19, name='n_epochs')
dim_n_convolutions = Integer(low=1, high=4, name='n_convolutions')
dim_dense_nodes = Integer(low=32, high=128, name='n_dense_nodes')


def spawn_network(args):
    """
    This function spawns a network with the hyperparameters specified by args and captures the output in output. After
    the network has been evaluated the accuracy of the network is captured by looking for the code "AxzcodeK" in output.
    :param args: a list of values for the dimensions in the order specified in optimizer declaration.
    :return: the accuracy the network achieved using the hyperparameters specified in args
    """
    time.sleep(5)
    point = args[0]
    gpu = args[1]
    cmd = "python3 DL_CNN_V09.py"
    cmd = cmd + " --learning_rate=" + str(point[0])
    cmd = cmd + " --n_epochs=" + str(point[1])
    cmd = cmd + " --n_convolutions=" + str(point[2])
    cmd = cmd + " --n_dense_nodes=" + str(point[3])
    cmd = cmd + " --gpu_id=" + str(gpu)
    cmd = cmd + " --n_runs=" + str(n_runs)

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p.communicate()[0]
    output = output.decode("utf-8")
    if "AxzcodeK" not in output:
        print(output)
        assert False, "Node Ended prematurely. Check output above."
    acc = output.split("AxzcodeK")[1]
    print("ACC was", acc)
    return float(acc)


def bayesian_optimize():
    """
    Creates a skopt Bayesian Optimizer, and calls it n_calls times. Each time the optimizer gives n_gpus points
    to evaluate. These are passed to n_gpus threads created through python multiprocessing. Each of these threads
    is assigned a point(p.map) and calls spawn_network with the point as the argument. Finally the optimization
    results are saved.
    :return: ---
    """
    start_time = time.time()
    optimizer = Optimizer(
        dimensions=[dim_learning_rate, dim_n_epochs, dim_n_convolutions, dim_dense_nodes],
        random_state=1
    )
    gpus = range(3, n_gpus+3)
    for i in range(1, n_calls+1):
        x = optimizer.ask(n_points=n_gpus)  # x is a list of n_points points
        p = Pool(n_gpus)
        to_map = list(zip(x, gpus))
        print("Points for iteration", i, ":")
        print(to_map)
        y = p.map(spawn_network, to_map)
        optimizer.tell(x, y)
        print("#" * 100)
        print("Optimization cylce", i, "done.")
        print("#" * 100)
    print("Best setup found:")
    print(min(optimizer.yi))  # print the best objective found
    sorted_sets = sorted(list(zip(optimizer.yi, optimizer.Xi)), key=lambda tup: tup[0])
    print("BEST SET:", sorted_sets[0])
    end_time = time.time()
    print("It took:", str(end_time-start_time), "seconds")
    try:
        file_path = "logs/dl_optimizer_result.txt"
        label_file = open(file_path, "w")
        label_file.write("Best setup found:\n")
        label_file.write(str(sorted_sets[0]))
        label_file.write("\nTime to process: ")
        label_file.write(str(end_time-start_time))
    finally:
        label_file.close()
    pickle.dump(sorted_sets, open("logs/optimizer_points.pkl", "wb"))


def main():
    bayesian_optimize()


if __name__ == "__main__":
    main()
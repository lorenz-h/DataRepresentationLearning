import os
import time
import subprocess
n_gpus = 4
system_gpus = 7
points = [[("batch_size", 256), ("n_epochs", 300)], [("batch_size", 128), ("n_epochs", 200)]]


def spawn_network(point, gpu):
    cmd_list = ["python3", "CF10_CNN_V01.py"]
    for option in point:
        cmd_list.append("--" + str(option[0]))
        cmd_list.append(str(option[1]))
    cmd_list.append("&")
    subprocess.call(cmd_list)
    print("Started Solve at point", point, "on gpu", gpu)


def feed_queue_into_gpus():
    while len(points) is not 0:
        time.sleep(5)
        gpus = check_available_gpus()
        spawn_network(points[0], gpus[0])
        points.pop(0)
    print("All points have been spawned")


def check_available_gpus():
    available_gpus = []
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            available_gpus.append(gpu)
    if len(available_gpus) is not 0:
        print("The following GPUs are available", available_gpus)
    else:
        print("All GPUs are busy.")
    return available_gpus


feed_queue_into_gpus()

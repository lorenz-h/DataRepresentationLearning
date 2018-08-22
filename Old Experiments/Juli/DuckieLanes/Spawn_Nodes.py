import time
import subprocess
import numpy as np

n_gpus = 4
system_gpus = 7


def generate_points():
    generated = []
    for i in range(10):
        point = [("batch_size", np.random.randint(1, 8)),
                 ("n_epochs", np.random.randint(1, 10)*5),
                 ("adam_learning_rate", np.random.randint(1, 10)*0.001)]
        generated.append(point)
    return generated


def check_available_gpus():
    system_gpus = 7
    available_gpus = []
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            available_gpus.append(gpu)
    return available_gpus


def spawn_network(point, gpu):
    cmd = "nohup python3 DL_CNN_V06.py"
    for option in point:
        option_str = " --" + str(option[0]) + "=" + str(option[1])
        cmd += option_str
    gpu_str = " --gpu_id="+str(gpu)
    cmd += gpu_str
    cmd += " &"
    print("EXEC:", cmd)
    p = subprocess.Popen(cmd, shell=True)
    print("Started Solve at point", point, "on GPU", gpu)


def feed_queue_into_gpus():

    while len(points) is not 0:
        gpus = check_available_gpus()
        while len(gpus) is not 0:
            if len(points) is 0:
                break
            spawn_network(points[0], gpus[0])
            time.sleep(2)
            points.pop(0)
            gpus.pop(0)
        print("Sleep. Points remaining:", len(points))

        time.sleep(60)
    print("All points have been spawned")


points = generate_points()
print(len(points), "points generated")

feed_queue_into_gpus()


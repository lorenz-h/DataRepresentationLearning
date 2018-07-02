import os
import time
n_gpus = 4


os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 MNIST_CNN_V03.py 0 &")
os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 MNIST_CNN_V03.py 1 &")
os.system("CUDA_VISIBLE_DEVICES=2 nohup python3 MNIST_CNN_V03.py 2 &")
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 MNIST_CNN_V03.py 3 &")

print("Spawned four Solvers")
time.sleep(1)
print("Starting Progress Indicator...")
try:
    i = 0
    proceed = 1
    points_to_calc = [3, 3, 3, 2]
    while proceed < n_gpus:
        i = i+1
        time.sleep(30)
        print("Status Update", i, ":")
        for gpu in range(n_gpus):
            file_string = "output_gpu" + str(gpu) + ".txt"
            thefile = open(file_string, 'r')
            progress = len(thefile.readlines())/2
            if progress == points_to_calc[gpu]:
                proceed = proceed + 1
            print("GPU", gpu, " has evaluated at :", progress, "Points so far")
            thefile.close()
    print("All Calculations done. Exiting Progress Indicator...")
    print("Total Time: ", i*30/60, " minutes")
except KeyboardInterrupt:
    print('\n Progress Indicator prematurely terminated!')
    os.system("killall python3")

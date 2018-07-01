import os
os.system("CUDA_VISIBLE_DEVICES=0 nohup python3 MNIST_CNN_V03.py 0 &")
os.system("CUDA_VISIBLE_DEVICES=1 nohup python3 MNIST_CNN_V03.py 1 &")
os.system("CUDA_VISIBLE_DEVICES=2 nohup python3 MNIST_CNN_V03.py 2 &")
os.system("CUDA_VISIBLE_DEVICES=3 nohup python3 MNIST_CNN_V03.py 3 &")

print("Spawned four Solvers")

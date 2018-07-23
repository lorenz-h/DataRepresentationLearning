import numpy as np
from scipy.misc import imread, imsave
from os import listdir
import matplotlib.pyplot as plt


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def create_file():
    raw_image_file_names = list_files("Dataset_V02", "png")
    raw_label_file_names = list_files("Dataset_V02", "txt")

    i = 0
    for file_name in raw_image_file_names:
        file_description = file_name[:-4]
        print(file_description)
        i += 1
        img = imread("Dataset_V02/" + file_name)
        img = img[160:, ...]
        decider = np.random.randint(0, 8)
        if decider > 5:
            path = "Dataset_V02/Testing/" + file_description
        else:
            path = "Dataset_V02/Training/" + file_description
        imsave(path+".png", img)

        try:
            f_old = open("Dataset_V02/" + file_description + ".txt", 'r')
            f_new = open(path + ".txt", "w")
            label = f_old.read()
            f_new.write(label)
        finally:
            f_old.close()
            f_new.close()




create_file()
import numpy as np
from scipy.misc import imread, imsave, imresize
from os import listdir
import scipy.fftpack


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def write_files():
    raw_image_file_names = list_files("../Dataset_V02/Training/", "png")
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = imread("../Dataset_V02/Training/" + file_name)
        img = imresize(img, [200, 400])
        imsave("../Resized_Dataset_V02/Training/" + file_name, img)
        if i % 20 == 0:
            print("image", str(i), "processed")


def main():
    write_files()


if __name__ == "__main__":
    main()
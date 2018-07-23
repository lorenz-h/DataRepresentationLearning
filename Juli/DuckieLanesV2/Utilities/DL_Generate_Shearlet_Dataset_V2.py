from scipy.misc import imread, imsave
from os import listdir
import pyshearlab
import pickle
import numpy as np

shearlet_system = None


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def grayscale(array):
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


def get_shearlet_system(sizex, sizey, j_max):
    global shearlet_system
    if shearlet_system is None:
        cached_systems = list_files("_cache", "pkl")
        cache_file_name = "200_200_2.pkl"
        if cache_file_name in cached_systems:
            try:
                pkl_file = open("_cache/" + cache_file_name, 'rb')
                shearlet_system = pickle.load(pkl_file)
            finally:
                pkl_file.close()
        else:
            shearlet_system = pyshearlab.SLgetShearletSystem2D(0, sizex, sizey, j_max)
            try:
                pkl_file = open("_cache/" + cache_file_name, 'wb')
                shearlet_system = pickle.dump(shearlet_system, pkl_file)
            finally:
                pkl_file.close()


def shearlet_preprocessing():

    return 0


def write_files():
    raw_image_file_names = list_files("../Dataset_V02/Training/", "png")
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = imread("../Dataset_V02/Training/" + file_name)
        get_shearlet_system(460, 640, 2)
        img = shearlet_preprocessing(img)
        imsave("Shearlet_Dataset_V02/Training/" + file_name, img)
        if i % 20 == 0:
            print("image", str(i), "processed")


def main():
    get_shearlet_system(200,200,2)


if __name__ == "__main__":
    main()

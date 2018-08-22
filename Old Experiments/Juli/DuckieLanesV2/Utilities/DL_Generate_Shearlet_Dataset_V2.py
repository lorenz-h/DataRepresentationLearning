from scipy.misc import imread, imsave
from os import listdir
import pyshearlab
import pickle
import numpy as np
import matplotlib.pyplot as plt

source_folder = "../Dataset_V02/Training/"
destination_folder = "../Shearlet_Dataset_V02_2/Training/"
j_max = 2


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def grayscale(array):
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


def pad_to_square(array):
    max_len = max(array.shape)
    min_len = min(array.shape)
    output = np.zeros([max_len, max_len])
    output[0:min_len, :] = array
    return output


def grab_image(file):
    img = imread(source_folder + file)
    img = grayscale(img)
    img = pad_to_square(img)
    img = img / np.amax(img)
    assert img.ndim == 2 and img.shape[0] == img.shape[1]
    return img


def get_shearlet_system(input_size):
    cached_systems = list_files("_cache/", "pkl")
    file_name = str(input_size)+"_"+str(j_max)+".pkl"
    if file_name in cached_systems:
        try:
            sh_file = open("_cache/"+file_name, "rb")
            shearlet_sys = pickle.load(sh_file)
        finally:
            sh_file.close()
        print("Using Cached Shearlet System.")
    else:
        print("No Cached System found. Generating System...")
        shearlet_sys = pyshearlab.SLgetShearletSystem2D(0, input_size, input_size, j_max)
        try:
            sh_file = open("_cache/"+file_name, "wb")
            shearlet_sys = pickle.dump(shearlet_sys, sh_file)
        finally:
            sh_file.close()
        print("Generated new Shearlet System")
    return shearlet_sys


def write_files(shear_sys, raw_image_file_names):
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = grab_image(file_name)
        coeffs = pyshearlab.SLsheardec2D(img, shear_sys)
        coeffs = coeffs[0:320, :]
        flt = np.mean(coeffs[..., 6:13], -1)
        flt = flt - np.amin(flt)
        flt = flt / np.amax(flt)
        imsave(destination_folder + file_name, flt)
        if i % 20 == 0:
            print("Image", i, "done.")


def main():
    raw_image_file_names = list_files(source_folder, "png")
    file_0 = next(raw_image_file_names)
    img0 = grab_image(file_0)
    input_shape = img0.shape[0]
    shearlet_system = get_shearlet_system(input_shape)
    write_files(shearlet_system, raw_image_file_names)


if __name__ == "__main__":
    main()

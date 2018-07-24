from scipy.misc import imread, imsave
from os import listdir
import pyshearlab
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pywt
from DL_Utilities import list_files, grayscale


source_folder = "../Dataset_V02/Testing/"
destination_folder = "../Wavelet_Dataset_V02/Testing/"
j_max = 2


def wavelet_and_crop(array):
    wp = pywt.wavedec2(array, 'db1', mode='symmetric', level=2)
    level1 = np.vstack((np.hstack((wp[0], wp[1][0])), np.hstack((wp[1][1], wp[1][2]))))
    level2 = np.vstack((np.hstack((level1, wp[2][0])), np.hstack((wp[2][1], wp[2][2]))))
    return level2


def grab_image(file):
    img = imread(source_folder + file)
    img = grayscale(img)
    img = img / np.amax(img)
    assert img.ndim == 2
    return img


def visualize():
    raw_image_file_names = list_files(source_folder, "png")
    for i in range(2):
        file_name = next(raw_image_file_names)
        img = grab_image(file_name)
        wavelet_coeffs = wavelet_and_crop(img)
        wavelet_coeffs = wavelet_coeffs - np.amin(wavelet_coeffs)
        wavelet_coeffs /= np.amax(wavelet_coeffs)
        print(np.mean(wavelet_coeffs))
        plt.imshow(wavelet_coeffs, cmap="gray")
        plt.show()


def write_files():
    raw_image_file_names = list_files(source_folder, "png")
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = grab_image(file_name)
        wavelet_coeffs = wavelet_and_crop(img)
        imsave(destination_folder + file_name, wavelet_coeffs)
        if i % 20 == 0:
            print("Image", i, "done.")


def main():
    visualize()


if __name__ == "__main__":
    main()

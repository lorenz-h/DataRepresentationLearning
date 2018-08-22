import numpy as np
from scipy.misc import imread, imsave
from os import listdir
import scipy.fftpack


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def dct2(image):
    assert image.ndim == 2
    output = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    return output


def grayscale(array):
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


def write_files():
    raw_image_file_names = list_files("../Dataset_V02/Testing/", "png")
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = imread("../Dataset_V02/Testing/" + file_name)
        img = grayscale(img)
        coeffs = dct2(img)
        coeffs = coeffs - np.amin(coeffs)
        coeffs = np.sqrt(coeffs)
        coeffs = coeffs / np.amax(coeffs)
        coeffs = coeffs[:200, :400]
        imsave("../DCT_Dataset_V02/Testing/" + file_name, coeffs)
        if i % 20 == 0:
            print("image", str(i), "processed")


def main():
    write_files()


if __name__ == "__main__":
    main()
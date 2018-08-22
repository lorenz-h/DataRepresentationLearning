from os import listdir
from scipy.misc import imsave, imread
import numpy as np
import pyshearlab
import pickle

shearlet_system = None


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def grayscale(array):
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


def shearlet_crop(array, shearsys):
    assert array.ndim == 2
    coeffs = pyshearlab.SLsheardec2D(array, shearsys)
    return coeffs


def pad_to_square(array):
    max_len = max(array.shape)
    min_len = min(array.shape)
    output = np.zeros([max_len, max_len])
    output[0:min_len, :] = array
    return output


def get_shearlet_system(sizex, sizey, j_max):
    """
    This function either loads a cached shearlet system or loads a system from disk or computes a new one and saves it to disk and memory.
    :param sizex:
    :param sizey:
    :param j_max:
    :return:
    """
    global shearlet_system
    if shearlet_system is None:
        shearlet_system = pyshearlab.SLgetShearletSystem2D(0, sizex, sizey, j_max)
    return shearlet_system


def full_wavelet_preprocessing(image):
    image = grayscale(image)
    if np.amax(image) > 2.0:
        image = image / np.amax(image)
    image = pad_to_square(image)
    img_size = image.shape[0]
    shearletsystem = get_shearlet_system(img_size, img_size, 2)
    coeffs = shearlet_crop(image, shearletsystem)
    coeffs = coeffs[160:480, ...]
    flt = coeffs[..., 6:13]
    flt = np.mean(flt, -1)
    return flt


def write_files():
    raw_image_file_names = list_files("Dataset2/Testing/", "png")
    i = 0
    for file_name in raw_image_file_names:
        i += 1
        img = imread("Dataset2/Testing/" + file_name)
        img = full_wavelet_preprocessing(img)
        imsave("Shearlet_Dataset/Testing/" + file_name, img)
        if i % 20 == 0:
            print("image", str(i), "processed")


def main():
    print("HELLO")
    

if __name__ == "__main__":
    main()


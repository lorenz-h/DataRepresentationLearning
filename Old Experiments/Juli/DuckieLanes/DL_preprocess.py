import pywt
import scipy.fftpack
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import pyshearlab
from os import listdir
import pickle


"""
def dct2(image):
    
    assert image.ndim == 2
    output = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    output = map_dct_coeffs(output)
    return output


def map_dct_coeffs(array):
    
    array = array-np.amin(array)
    array = np.sqrt(array)
    normalized = array / (np.amax(array) * 0.5)
    return normalized


def wavelet(image):
    wp = pywt.wavedec2(image, 'db1', mode='symmetric', level=2)
    level1 = np.vstack((np.hstack((wp[0], wp[1][0])), np.hstack((wp[1][1], wp[1][2]))))
    level2 = np.vstack((np.hstack((level1, wp[2][0])), np.hstack((wp[2][1], wp[2][2]))))
    return level2
"""


shear_sys_in_memory = None


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
    global shear_sys_in_memory

    if shear_sys_in_memory is None:
        file_name = str(sizex)+"_"+str(sizey)+"_"+str(j_max)+".pickle"
        if file_name in listdir("Shearlet_Systems"):
            print("Spectra Already cached. Loading file...")
            with open("Shearlet_Systems/" + file_name, 'rb') as handle:
                shear_sys = pickle.load(handle)
            print("Successfully loaded Shearlet System")
        else:
            print("Computing Shearlet Spectra...")
            shear_sys = pyshearlab.SLgetShearletSystem2D(0, sizex, sizey, j_max)
            with open("Shearlet_Systems/" + file_name, 'wb') as handle:
                pickle.dump(shear_sys, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Cached Shearlet Spectra under "+file_name)
        shear_sys_in_memory = shear_sys
    return shear_sys_in_memory


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


def numpy_preprocess(image):
    output = full_wavelet_preprocessing(image)
    output = output.astype(dtype="float32")
    return output


def main():
    file_string = "Dataset2/Training/sample" + str(120) + ".png"
    image = scipy.misc.imread(file_string).astype(float)
    coeffs = full_wavelet_preprocessing(image)
    if coeffs.ndim == 3:
        fig = plt.figure(figsize=(40, 40))
        for i in range(17):
            ax = fig.add_subplot(5, 5, i + 1)
            ax.imshow(coeffs[..., i], origin='upper', cmap=plt.cm.gray)
            ax.set_title(str(i), fontsize=12)
        flt = coeffs[..., 6:13]
        flt = np.mean(flt, -1)
        ax = fig.add_subplot(5, 5, 19)
        ax.imshow(flt, origin='upper', cmap=plt.cm.gray)
        plt.show()
    if coeffs.ndim == 2:
        plt.imshow(coeffs, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()

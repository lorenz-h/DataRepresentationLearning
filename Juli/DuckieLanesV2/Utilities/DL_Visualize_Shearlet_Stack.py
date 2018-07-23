import numpy as np
import pyshearlab
import scipy.misc
import matplotlib.pyplot as plt

shearlet_system = None


def greyscale(array):
    assert array.ndim == 3
    array = np.mean(array, 2)
    return array


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


def wavelet_crop(array, shearsys):
    coeffs_ = pyshearlab.SLsheardec2D(array, shearsys)
    print(coeffs_.shape)
    coeffs_ = coeffs_[:320, ...]
    return coeffs_


def show_coeffs_stack():
    global shearlet_system
    file_string = "../Dataset_V02/Training/sample" + str(21) + ".png"
    image = scipy.misc.imread(file_string).astype(float)
    image = greyscale(image)
    image = image / 255
    image = pad_to_square(image)
    shearletsystem = get_shearlet_system(image.shape[0], image.shape[1], 2)
    coeffs = wavelet_crop(image, shearletsystem)
    fig = plt.figure(figsize=(20, 20))
    for i in range(17):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(coeffs[..., i], origin='upper', cmap=plt.cm.gray)
        ax.set_title(str(i), fontsize=12)
    plt.show()


def show_shearlets():
    global shearlet_system
    file_string = "../Dataset_V02/Training/sample" + str(21) + ".png"
    image = scipy.misc.imread(file_string).astype(float)
    image = greyscale(image)
    shearletsystem = get_shearlet_system(320, 320, 2)
    print(shearletsystem.keys())

    slts = shearletsystem["shearlets"]
    print(np.amin(slts))
    print(np.amax(slts))
    fig = plt.figure(figsize=(20, 20))
    for i in range(17):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(slts[..., i], origin='upper', cmap=plt.cm.gray)
        ax.set_title(str(i), fontsize=12)
    plt.show()


def main():
    show_coeffs_stack()


if __name__ == "__main__":
    main()

import scipy.misc
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt


n_features = 1
dct_shape = [190, 190]


def normalize(array):
    min_val = np.amin(array)
    array = array - min_val
    max_val = np.amax(array)
    array = array / max_val
    return array


def rgb_map(array):
    min_val = np.amin(array)
    array = array - min_val
    max_val = np.amax(array)
    array = array * 255 / max_val
    return array


def show_histogram(array, title):
    array = array.flatten()
    plt.hist(array, bins=[x/255 for x in range(0, 255)])
    plt.title(title)
    plt.show()


def dct2(image):
    assert image.ndim == 2
    return scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(image):
    assert image.ndim == 2
    return scipy.fftpack.idct(scipy.fftpack.idct(image, axis=0, norm='ortho'), axis=1, norm='ortho')


def main():
    for feature in range(0, n_features):
        file_string = "Dataset2/Training/sample" + str(feature) + ".png"
        image = scipy.misc.imread(file_string).astype(float)
        print(np.amax(image))
        image = image[..., 0]
        plt.imshow(image, cmap="gray")
        plt.show()
        dct = dct2(image)

        tiny = dct[0:90, 0:90]
        plt.imshow(tiny, cmap='gray')
        plt.title("8x8 DCTs of the image")
        plt.show()

        dct_recovered = np.zeros_like(image)
        dct_recovered[0:90, 0:90] = tiny
        image = idct2(dct_recovered)
        print(np.amax(image))
        plt.imshow(image, cmap="gray")
        plt.show()

        image_subsampled = image[0:480:4, 0:640:4]
        print(image_subsampled.shape)
        sample_recovered = scipy.misc.imresize(image_subsampled, (480, 640),interp="cubic")
        plt.imshow(sample_recovered, cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()

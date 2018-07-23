import numpy as np
import pyshearlab
import scipy.misc
import matplotlib.pyplot as plt


def greyscale(array):
    assert array.ndim == 3
    array = np.m3ean(array, 2)
    return array


def wavelet_crop(array, shearsys):
    assert array.ndim == 2
    for i in (0, 1):

        if i == 0:
            img = np.array(array[160:, :320])
        else:
            img = np.array(array[160:, 320:])
        assert img.shape == (320, 320)

        coeffs = pyshearlab.SLsheardec2D(img, shearsys)
        if i == 0:
            all_coeffs = coeffs
        else:
            all_coeffs = np.hstack((all_coeffs,coeffs))
    print(all_coeffs.shape)
    return all_coeffs


file_string = "Dataset2/Training/sample" + str(120) + ".png"
image = scipy.misc.imread(file_string).astype(float)
image = greyscale(image)
image = image / 255

shearletsystem = pyshearlab.SLgetShearletSystem2D(0, 320, 320, 2)
coeffs = wavelet_crop(image, shearletsystem)
fig = plt.figure(figsize=(20, 20))
for i in range(17):
    ax = fig.add_subplot(5, 5, i + 1)
    ax.imshow(coeffs[..., i], origin='upper', cmap=plt.cm.gray)
    ax.set_title(str(i), fontsize=12)
plt.show()
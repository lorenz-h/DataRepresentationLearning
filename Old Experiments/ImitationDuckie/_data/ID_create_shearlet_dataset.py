import csv
from scipy.misc import imread, imsave
from os import listdir
import pyshearlab
import pickle
import numpy as np

from _utils.ID_utils import yes_or_no

purpose = "evaluation"

destination_folder = "/media/sdb/hetzell/shearlet_dataset/" + purpose + "/"
j_max = 2
input_shape = [60, 80, 1]


def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))


def grayscale(array):
    assert array.ndim == 1
    array = np.mean(array, 2)
    return array


def pad_to_square(array):
    max_len = max(array.shape)
    min_len = min(array.shape)
    output = np.zeros([max_len, max_len])
    output[0:min_len, :] = array
    return output


def grab_image(file):
    image = imread(file)
    assert image is not None
    assert image.ndim == 2
    image = pad_to_square(image)
    image = image - np.amin(image)
    image = image / np.amax(image)
    assert image.ndim == 2 and image.shape[0] == image.shape[1]
    return image


def main():
    proceed = yes_or_no("Are you sure you would like to proceed this will potentially corrupt the datasets?")
    if not proceed:
        return 0
    print("Improved Version")
    input_size = 80
    shearlet_sys = pyshearlab.SLgetShearletSystem2D(0, input_size, input_size, j_max)
    rows = []
    with open('../_data/hetzell_raw_'+purpose+'_data.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            rows.append(row)
        print(rows.__len__())
    new_rows = []
    i = 0
    for row in rows:
        img = grab_image(row[1])
        coeffs = pyshearlab.SLsheardec2D(img, shearlet_sys)
        coeffs = coeffs[0:60, :]
        flt = np.mean(coeffs[..., 6:13], -1)
        flt = flt - np.amin(flt)
        flt = flt / np.amax(flt)
        file_name = "sample" + str(i) + ".png"
        destination_path = destination_folder + file_name
        imsave(destination_path, flt)
        if i == 0:
            print("SHAPE", coeffs.shape)
        if i % 20 == 0:
            print("Image", i, "done.")
        new_row = [row[0], destination_path]
        new_rows.append(new_row)
        i += 1
    with open('../_data/hetzell_shearlet_'+purpose+'_data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in new_rows:
            spamwriter.writerow(row)


if __name__ == "__main__":
    main()


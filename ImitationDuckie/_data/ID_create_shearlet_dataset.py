import csv
from scipy.misc import imread, imsave
from os import listdir
import pyshearlab
import pickle
import numpy as np


destination_folder = "/media/sdb/hetzell/shearlet_dataset/testing/"
j_max = 2
input_shape = [480, 640, 3]


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
    image = imread(file)
    assert image is not None
    image = grayscale(image)
    image = pad_to_square(image)
    image = image / np.amax(image)
    assert image.ndim == 2 and image.shape[0] == image.shape[1]
    return image


def get_shearlet_system(input_size):
    cached_systems = list_files("../_cache/", "pkl")
    file_name = str(input_size)+"_"+str(j_max)+".pkl"
    if file_name in cached_systems:
        try:
            sh_file = open("../_cache/"+file_name, "rb")
            shearlet_sys = pickle.load(sh_file)
        finally:
            sh_file.close()
        print("Using Cached Shearlet System.")
    else:
        print("No Cached System found. Generating System...")
        shearlet_sys = pyshearlab.SLgetShearletSystem2D(0, input_size, input_size, j_max)
        try:
            sh_file = open("../_cache/" + file_name, "wb")
            shearlet_sys = pickle.dump(shearlet_sys, sh_file)
        finally:
            sh_file.close()
        print("Generated new Shearlet System")
    return shearlet_sys


shearlet_system = get_shearlet_system(640)
rows = []
new_rows = []
with open('../_data/hetzell_testing_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        rows.append(row)
    print(rows.__len__())

i = 0
for row in rows:
    img = grab_image(row[1])
    coeffs = pyshearlab.SLsheardec2D(img, shearlet_system)
    coeffs = coeffs[0:480, :]
    flt = np.mean(coeffs[..., 6:13], -1)
    flt = flt - np.amin(flt)
    flt = flt / np.amax(flt)
    file_name = "sample" + str(i) + ".png"
    destination_path = destination_folder + file_name
    imsave(destination_path, flt)
    if i % 20 == 0:
        print("Image", i, "done.")
    new_row = [row[0], destination_path]
    new_rows.append(new_row)
    i += 1
with open('../_data/hetzell_shearlet_testing_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in new_rows:
        spamwriter.writerow(row)

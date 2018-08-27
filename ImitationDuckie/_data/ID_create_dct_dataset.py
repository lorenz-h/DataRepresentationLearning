import csv
from scipy.misc import imsave, imread
import numpy as np
import scipy.fftpack
destination_folder = "/media/sdb/hetzell/dct_dataset/training/"
j_max = 2
input_shape = [480, 640, 3]

rows = []
new_rows = []


def grab_image(file):
    image = imread(file)
    assert image.ndim == 2
    assert image is not None
    image = image / np.amax(image)
    return image


def dct2(image):
    assert image.ndim == 2
    output = scipy.fftpack.dct(scipy.fftpack.dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    return output


with open('../_data/hetzell_raw_training_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        rows.append(row)
    print(rows.__len__())

i = 0
for row in rows:
    img = grab_image(row[1])
    crop = dct2(img)
    crop = crop - np.amin(crop)
    crop = crop / np.amax(crop)

    file_name = "sample" + str(i) + ".png"
    destination_path = destination_folder + file_name
    imsave(destination_path, crop)
    if i % 20 == 0:
        print("Image", i, "done.")
    new_row = [row[0], destination_path]
    new_rows.append(new_row)
    i += 1
with open('../_data/hetzell_dct_training_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in new_rows:
        spamwriter.writerow(row)

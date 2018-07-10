import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.misc import imsave


def parse_dataset():
    images = np.load("x.npy", encoding="bytes")
    print(images.shape)
    labels = np.load("y.npy", encoding="bytes")

    sorted_labels = sorted(labels, key=lambda tup: tup[1])
    sorted_images = sorted(images, key=lambda tup: tup[1])

    dataset = []
    for i in range(sorted_images.__len__()):
        desired_time = sorted_images[i][1]
        best_matched_label = min(sorted_labels, key=lambda tup: abs(tup[1] - desired_time))
        if abs(best_matched_label[1] - desired_time) < 50000000:  # deltaTmax = 0.1 seconds
            dataset.append((sorted_images[i][0], best_matched_label[0]))
    print(dataset.__len__())

    dataset_array = np.array(dataset)
    images = np.array(dataset_array[:, 0].tolist())
    labels = dataset_array[:, 1].tolist()
    print(images.shape)
    for i in range(0, int(images.shape[0]*0.8)):
        image = images[i, ...]
        file_string = "Dataset/Training/sample"+str(i)+".png"
        imsave(file_string, image)
        label_string = "Dataset/Training/sample"+str(i)+".txt"
        file = open(label_string, "w")
        file.write(str(labels[i]))
        file.close()
        if i % 50 == 0:
            print("Storing Sample"+str(i))

    for i in range(int(images.shape[0]*0.8), images.shape[0]):
        image = images[i, ...]
        file_string = "Dataset/Testing/sample"+str(i)+".png"
        imsave(file_string, image)
        label_string = "Dataset/Testing/sample" + str(i) + ".txt"
        file = open(label_string, "w")
        file.write(str(labels[i]))
        file.close()
        if i % 50 == 0:
            print("Storing Sample"+str(i))


def main():
    parse_dataset()


if __name__ == "__main__":
    main()
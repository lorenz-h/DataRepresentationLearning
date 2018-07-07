import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


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
        if abs(best_matched_label[1] - desired_time) < 100000000:  # deltaTmax = 0.1 seconds
            dataset.append((sorted_images[i][0], best_matched_label[0]))
    print(dataset.__len__())

    shuffle(dataset)
    dataset_array = np.array(dataset)
    images = np.array(dataset_array[:, 0].tolist(), dtype="float16")
    labels = np.array(dataset_array[:, 1].tolist(), dtype="float32")
    print(images.shape)
    print(labels.shape)
    training_data = images[0:int(images.shape[0]*0.8), ...]
    training_labels = images[0:int(labels.shape[0] * 0.8), ...]
    testing_data = images[int(images.shape[0]*0.8):, ...]
    testing_labels = images[int(labels.shape[0] * 0.8):, ...]

    return training_data, training_labels, testing_data, testing_labels


def main():
    training_data, x, y, z = parse_dataset()


if __name__ == "__main__":
    main()
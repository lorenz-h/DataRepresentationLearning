import pickle
import os
import numpy as np
from scipy.fftpack import dct


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def process_dictionary(dictionary):
    images = dictionary[b'data']
    return images, labels


def reshape_batch(input_batch):
    red = input_batch[:, :1024]
    green = input_batch[:, 1024:2048]
    blue = input_batch[:, 2048:]
    red = np.reshape(red, [input_batch.shape[0], 32, 32, ])
    green = np.reshape(green, [input_batch.shape[0], 32, 32, ])
    blue = np.reshape(blue, [input_batch.shape[0], 32, 32, ])
    for i in range(input_batch.shape[0]):
        image_r = red[i, ...]
        image_g = green[i, ...]
        image_b = blue[i, ...]
        red[i, ...] = preprocess(image_r)
        green[i, ...] = preprocess(image_g)
        blue[i, ...] = preprocess(image_b)
    output = np.stack([red, green, blue], 3)
    return output


def preprocess(array):
    # output = dct(dct(array.T).T)
    output = array
    return output


def import_testing_data():
    file_path = os.path.join("cifar-10-batches-py", "test_batch")
    dictionary = unpickle(file_path)
    images = dictionary[b'data']
    labels = dictionary[b'labels']
    images = reshape_batch(images)
    images = images.astype('float32')
    print("Successfully imported testing data")
    return images, labels


def import_training_data():
    for i in range(1, 6):
        file_name = "data_batch_"+str(i)
        file_path = os.path.join("cifar-10-batches-py", file_name)
        dictionary = unpickle(file_path)
        if i == 1:
            images = dictionary[b'data']
            labels = dictionary[b'labels']
        else:
            images = np.vstack((images, dictionary[b'data']))
            labels = np.hstack((labels, dictionary[b'labels']))
    images = reshape_batch(images)
    images = images.astype('float32')
    print(images.shape)
    print("Successfully imported training data")
    return images, labels


def main():
    import_training_data()
    import_testing_data()


if __name__ == "__main__":
    main()

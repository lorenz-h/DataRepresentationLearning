import tensorflow as tf

# Describe the dataset:
columns = ['angularz', 'path_to_feature']
field_defaults = [[0.0], [""]]


def _parse_csv_line(line):
    """
    This reads a csv file.
    :param line: The line of the csv file to read.
    :return:
    """
    fields = tf.decode_csv(line, field_defaults)
    features = dict(zip(columns, fields))
    label = features.pop("angularz")
    features = features.pop("path_to_feature")
    return features, label


def _parse_image(image_path, label):
    """
    This maps the image path in the csv file to the acutal image file
    :param image_path: the path to the image
    :param label: the label for the image
    :return: the image array and the corrsponding label
    """
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)
    img_decoded = tf.cast(img_decoded / 255, dtype=tf.float32)
    return img_decoded, label


def create_dataset(batch_size, csv_file):
    """
    This creates a tensorflow dataset object.
    :param batch_size: the batch size to use
    :param csv_file: the csv file to read the dataset from
    :return:
    """
    ds = tf.data.TextLineDataset(csv_file)
    ds = ds.map(_parse_csv_line, num_parallel_calls=4)
    ds = ds.map(_parse_image, num_parallel_calls=16)
    ds = ds.shuffle(buffer_size=64)
    ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))
    ds = ds.prefetch(buffer_size=batch_size * 2)
    return ds


def main():
    print("Not for standalone execution")


if __name__ == "__main__":
    main()

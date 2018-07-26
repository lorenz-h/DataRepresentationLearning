import tensorflow as tf

# Describe the dataset:
columns = ['angularz', 'path_to_feature']
field_defaults = [[0.0], [""]]


def _parse_csv_line(line):
    fields = tf.decode_csv(line, field_defaults)
    features = dict(zip(columns, fields))
    label = features.pop("angularz")
    features = features.pop("path_to_feature")
    return features, label


def _parse_image(image_path, label):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)
    img_decoded = tf.cast(img_decoded / 255, dtype=tf.float32)
    return img_decoded, label


def create_dataset(batch_size, csv_file):
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

import tensorflow as tf


def grab_files(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=input_shape[2])
    img_decoded = tf.cast(img_decoded / 255, dtype=tf.float32)
    label_file = tf.read_file(label_path)
    label = tf.string_to_number(label_file, out_type=tf.float64)
    return img_decoded, label


def flip(image, label):
    """
    :param image: the image to be flipped as a tf.Tensor
    :param label: the label associated with the unflipped image
    :return:
    """
    img = tf.image.flip_left_right(image)
    label = tf.negative(label)
    return img, label


def map_labels(image, label):
    """
    this recenters the label around 0
    :param image:
    :param label:
    :return:
    """
    label = tf.add(label, 1.0)
    label = tf.multiply(label, 10.0)
    label = tf.cast(label, dtype=tf.float32)
    noise = tf.multiply(tf.random_normal(shape=tf.shape(label), mean=0.0, stddev=0.6, dtype=tf.float32), 0.005)
    label = label + noise
    return image, label


def create_dataset(evaluation, batch_size, dataset_folder, input_shape_pass):
    """
    This creates a tf.data.dataset
    :param input_shape_pass: the shape of the input
    :param evaluation: bool, True if the dataset is to be used for evaluation
    :param batch_size: the size of batches into which to divide the dataset
    :param dataset_folder: the folder from where the data should be loaded
    :return:
    """
    global input_shape
    input_shape = input_shape_pass
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files(dataset_folder+"/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files(dataset_folder+"/"+test_train+"/*.txt", shuffle=False)
    normal = tf.data.Dataset.zip((images, labels))
    with tf.device('/cpu:0'):
        normal = normal.map(map_func=grab_files, num_parallel_calls=12)
        if not evaluation:
            flipped = normal.map(map_func=flip, num_parallel_calls=6)
            normal = normal.concatenate(flipped)
    dataset = normal.map(map_func=map_labels)
    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size=batch_size))
    dataset = dataset.prefetch(buffer_size=batch_size*2)
    return dataset

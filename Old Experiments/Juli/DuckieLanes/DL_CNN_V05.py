import tensorflow as tf
import argparse
import os
from scipy.misc import imsave, imread
import subprocess

from DL_preprocess import numpy_preprocess


batch_size = 64
processing_tog = True
shuffle_buffer_size = batch_size*2
prefetch_buffer_size = batch_size*2
adam_learning_rate = 0.01
n_epochs = 25
n_evaluations = 4
logging = False

gpu_id = 0


def convolution(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def cnn(x):
    x = tf.reshape(x, shape=[-1, 320, 640, 1])

    w_conv1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
    b_conv1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(convolution(x, w_conv1) + b_conv1)
    conv1 = maxpool2d(conv1)

    w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    b_conv2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(convolution(conv1, w_conv2) + b_conv2)
    conv2 = maxpool2d(conv2)
    conv2_shape = [conv2.shape[1].value, conv2.shape[2].value, conv2.shape[3].value]
    w_fc = tf.Variable(tf.random_normal([conv2_shape[0] * conv2_shape[1] * conv2_shape[2], 64]))
    b_fc = tf.Variable(tf.random_normal([64]))
    fc = tf.reshape(conv2, [-1, conv2_shape[0] * conv2_shape[1] * conv2_shape[2]])
    fc = tf.matmul(fc, w_fc) + b_fc

    w_out = tf.Variable(tf.random_normal([64, 1]))
    b_out = tf.Variable(tf.random_normal([1]))

    output = tf.matmul(fc, w_out) + b_out
    return output


def open_label(file_path):
    """
    :param file_path: path to label text file
    :return: label as float
    """
    label_file = open(file_path, "r")
    try:
        label = label_file.read()
    finally:
        label_file.close()
    label = float(label)
    return label


def grab_files(image_path, label_path):
    img_file = tf.read_file(image_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    img_decoded = tf.cast(img_decoded / 255, dtype=tf.float32)
    label = tf.py_func(open_label, [label_path], "double")
    label = label * 10.0
    return img_decoded, label


def normal_parse_fn(image_path, label_path):
    image, label = grab_files(image_path, label_path)
    if processing_tog:
        image = tf.py_func(numpy_preprocess, [image], tf.float32)
    return image, label


def flipped_parse_fn(image_path, label_path):
    image, label = grab_files(image_path, label_path)
    image = tf.image.flip_left_right(image)
    if processing_tog:
        image = tf.py_func(numpy_preprocess, [image], tf.float32)
    return image, label


def create_dataset(evaluation):
    if evaluation:
        test_train = "Testing"
    else:
        test_train = "Training"
    images = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.png", shuffle=False)
    labels = tf.data.Dataset.list_files("Dataset2/"+test_train+"/*.txt", shuffle=False)
    normal = tf.data.Dataset.zip((images, labels))
    flipped = normal.take(-1)

    normal = normal.map(map_func=normal_parse_fn)
    flipped = flipped.map(map_func=flipped_parse_fn)
    dataset = normal.concatenate(flipped)

    dataset = dataset.shuffle(buffer_size=batch_size)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset


def setup_network():
    tr_data = create_dataset(evaluation=False)
    val_data = create_dataset(evaluation=True)

    iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
    next_features, next_labels = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data)
    validation_init_op = iterator.make_initializer(val_data)

    prediction = cnn(next_features)
    loss = tf.cast(tf.losses.absolute_difference(labels=next_labels, predictions=prediction), dtype=tf.float32)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=adam_learning_rate).minimize(loss)

    accuracy = tf.cast(tf.losses.absolute_difference(labels=next_labels, predictions=prediction), dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        if logging:
            gpu_logdir = "gpu" + str(gpu_id) + "processing" + str(processing_tog)
            train_logdir = os.path.join("logs", gpu_logdir, "train")
            test_logdir = os.path.join("logs", gpu_logdir, "test")
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

        sess.run(init_op)
        i = 0
        print("Starting Training...")
        for epoch in range(1, n_epochs+1):
            sess.run(training_init_op)
            epoch_loss = 0
            epoch_acc = 0
            batches = 0
            while True:
                try:
                    if logging:
                        summary, lss, _, acc = sess.run([merged, loss, optimizer, accuracy])
                        train_writer.add_summary(summary, i)
                        i += 1
                    else:
                        lss, _, acc = sess.run([loss, optimizer, accuracy])
                    epoch_loss += lss
                    epoch_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            epoch_loss = epoch_loss/batches/batch_size
            epoch_acc = epoch_acc/batches/batch_size
            print("Finished Epoch", epoch, "- Training Loss:", epoch_loss, "- Accuracy:", epoch_acc)
        print("Starting Evaluation...")
        for ev in range(n_evaluations):
            sess.run(validation_init_op)
            eval_acc = 0
            batches = 0
            while True:
                try:
                    images, acc = sess.run([next_features, loss])
                    if batches < 5:
                        image = images[0, ...]
                        imsave(str(batches)+".png", image)
                    eval_acc += acc
                    batches += 1
                except tf.errors.OutOfRangeError:
                    break
            print("Evaluation", ev, "done.")
        eval_acc = eval_acc/batches/batch_size
        print("Average Accuracy over", n_evaluations, "was", eval_acc)


class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[32m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def str2bool(v):
    # this is a neccesary workaround to ensure that bools will be parsed correctly
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=type(globals()["gpu_id"]))
    parser.add_argument("--batch_size", type=type(globals()["batch_size"]))
    parser.add_argument("--adam_learning_rate", type=type(globals()["adam_learning_rate"]))
    parser.add_argument("--n_epochs", type=type(globals()["n_epochs"]))
    parser.add_argument("--logging", type=str2bool)
    parser.add_argument("--processing_tog", type=str2bool)

    args = parser.parse_args()

    print("#"*100, "\n")
    print("Arguments:")
    for argument in args.__dict__:
        if args.__dict__[argument] is None:
            print(Colors.WARNING, argument, "argument not given... Falling back to default value of: ",
                  globals()[argument], Colors.ENDC)
        else:
            globals()[argument] = getattr(args, argument)
            print(Colors.OKGREEN, argument, "argument parsed successfully as", globals()[argument], Colors.ENDC)

    print("")
    print("#" * 100)


def check_available_gpus():
    system_gpus = 7
    for gpu in range(system_gpus):
        command_str = "(nvidia-smi --id=" + str(gpu) + ")"
        result = subprocess.run(command_str, shell=True, stdout=subprocess.PIPE)
        if "No running processes found" in result.stdout.decode("utf-8"):
            return gpu
    assert False, "All GPUs are currently busy."


def check_preprocessing_pipeline():
    file_string = "Dataset2/Training/sample" + str(120) + ".png"
    image = imread(file_string).astype(float)
    output = numpy_preprocess(image)
    return output.shape


def main():
    global gpu_id
    global input_shape

    gpu_id = check_available_gpus()
    print("GPU", gpu_id, "is free and will be used.")
    parse_arguments()

    input_shape = check_preprocessing_pipeline()
    print("Preprocessing generates array with shape:"+str(input_shape))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    setup_network()


if __name__ == "__main__":
    main()


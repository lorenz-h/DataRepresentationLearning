import tensorflow as tf
from DL_Input_Pipeline import create_dataset
from DL_Utilities import get_input_shape
import numpy as np
import matplotlib.pyplot as plt

dataset_folder = "../Dataset_V02"
input_shape = get_input_shape(dataset_folder)
print("Input Shape:", input_shape)
tr_data = create_dataset(True, 128, dataset_folder, input_shape)
iterator = tf.data.Iterator.from_structure(tr_data.output_types, tr_data.output_shapes)
next_features, next_labels = iterator.get_next()
training_init_op = iterator.make_initializer(tr_data)

mean, var = tf.nn.moments(next_labels, axes=[0])
std = tf.sqrt(var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)
    avg_label = 0
    avg_std = 0
    batches = 0
    labels = []
    while True:
        try:
            l, m, s = sess.run([next_labels, mean, std])
            avg_label += m
            avg_std += s
            batches += 1
            labels.append(l)
            if batches % 10 == 0:
                print("Batch", batches, "done.")
        except tf.errors.OutOfRangeError:
            break
    avg_label /= batches
    avg_std /= batches
    labels = np.array(labels)
    labels = labels.flatten()
    print(labels.shape)
    print(avg_label, "std:", avg_std)

    fig, ax = plt.subplots(1)
    mu = labels.mean()
    median = np.median(labels)
    sigma = labels.std()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mu, median, sigma)
    n, bins, patches = plt.hist(labels, 50, density=True, facecolor='g', alpha=1.0)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.show()

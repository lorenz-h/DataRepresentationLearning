import tensorflow as tf
from DL_Input_Pipeline import create_dataset
from DL_Utilities import get_input_shape, classify, gen_bins
import numpy as np
import matplotlib.pyplot as plt

dataset_folder = "Dataset_V02"
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
    labels = []
    ll = sess.run(next_labels)
    ll = sess.run(next_labels)
    ll = sess.run(next_labels)
    labels.append(ll)
labels = np.array(labels)
labels = labels.flatten()
bin_list = gen_bins()
for i in range(labels.size):
    labels[i] = classify(labels[i], bin_list)


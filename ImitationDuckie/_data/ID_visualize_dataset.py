import csv

import matplotlib.pyplot as plt
import numpy as np

train_rows = []
with open('../_data/hetzell_training_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        train_rows.append(row)
    print(train_rows.__len__())


test_rows = []
with open('../_data/hetzell_testing_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        test_rows.append(row)
    print(test_rows.__len__())

eval_rows = []
with open('../_data/hetzell_evaluation_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        eval_rows.append(row)
    print(eval_rows.__len__())


train_labels = [float(row[0]) for row in train_rows]
eval_labels = [float(row[0]) for row in eval_rows]
test_labels = [float(row[0]) for row in test_rows]

print("STANDARD DEVIATIONS:")
print("Train", np.std(train_labels))
print("Eval", np.std(eval_labels))
print("Test", np.std(test_labels))

print("Minimum Value:")
print("Train", np.min(train_labels))
print("Eval", np.min(eval_labels))
print("Test", np.min(test_labels))

print("Maximum Value:")
print("Train", np.max(train_labels))
print("Eval", np.max(eval_labels))
print("Test", np.max(test_labels))

bins = np.linspace(-1, 1, 100)

plt.hist(train_labels, bins, alpha=0.3, label='train_labels', density=True)
plt.hist(eval_labels, bins, alpha=0.3, label='eval_labels', density=True)
plt.hist(test_labels, bins, alpha=0.3, label='test_labels', density=True)
plt.legend(loc='upper right')

plt.show()

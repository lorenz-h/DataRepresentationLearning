import csv
import numpy as np
import matplotlib.pyplot as plt
"""
This prints and illustrates some interesting facts about the labels stored in the cvs files in csv_files.
"""

csv_files = ("hetzell_training_data.csv", "hetzell_evaluation_data.csv", "hetzell_testing_data.csv")

rows = []
for file in csv_files:
    with open('../_data/'+file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            rows.append(row[0])

labels = np.array(rows, dtype=float)
mean = np.mean(labels)
error = np.abs(labels - mean)
mean_static_error = np.mean(error)
variance = np.mean(np.square(error))
std = np.sqrt(variance)
print("Mean:", mean)
print("Mean static error:", mean_static_error)
print("Variance:", variance)
print("Standard Deviation", std)

plt.title("Absolute Label Size")
bins = [x / 200 for x in range(0, 200)]
plt.hist(np.abs(labels), bins=bins)
plt.show()


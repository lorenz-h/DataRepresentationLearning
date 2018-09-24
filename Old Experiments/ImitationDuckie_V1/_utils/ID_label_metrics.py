import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
This prints and illustrates some interesting facts about the labels stored in the cvs files in csv_files.
"""
scale = 1.25
my_dpi = 200

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
varianceLabels = np.mean(np.square(labels))
std = np.sqrt(variance)
print("Mean:", mean)
print("Mean static error:", mean_static_error)
print("Variance:", variance)
print("VarianceLabels:", varianceLabels)
print("Standard Deviation", std)

fig, ax = plt.subplots(figsize=(6/scale, 4/scale), dpi=my_dpi)
bins = [x / 200 for x in range(0, 200)]
plt.hist(np.abs(labels), bins=bins, density=True)
ax.set_ylabel("frequency in \%")
ax.set_xlabel(r'$\| \omega \|$')
fig.show()


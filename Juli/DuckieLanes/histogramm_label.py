import os
import matplotlib.pyplot as plt
import numpy as np

labels = []
for file in os.listdir("Dataset2/Training"):
    if file.endswith(".txt"):
        f = open("Dataset2/Training/"+file, 'r')
        try:
            label = f.read()
            labels.append(float(label))
        finally:
            f.close()
arr_labels = np.array(labels)
print(arr_labels.shape)
plt.hist(arr_labels, bins=[x/100 for x in range(-20, 20)])
plt.show()
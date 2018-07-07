import numpy as np
from scipy.misc import imsave

images = np.load("x.npy", encoding="bytes")
print(images.shape)
labels = np.load("y.npy", encoding="bytes")

sorted_labels = sorted(labels, key=lambda tup: tup[1])
sorted_images = sorted(images, key=lambda tup: tup[1])

dataset = []
for i in range(sorted_images.__len__()):
    desired_time = sorted_images[i][1]
    best_matched_label = min(sorted_labels, key=lambda tup: abs(tup[1] - desired_time))
    if abs(best_matched_label[1] - desired_time) < 100000000:  # deltaTmax = 0.1 seconds
        dataset.append((sorted_images[i][0], best_matched_label[0]))
print(dataset.__len__())

print(dataset[0][0].shape)
file_string = "Dataset/Training/sample"+str(0)+".png"
imsave(file_string, dataset[0][0])


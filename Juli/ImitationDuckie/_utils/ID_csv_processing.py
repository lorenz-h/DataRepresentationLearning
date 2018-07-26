import csv
import random

"""
This takes Aurels csv files, reads both of them into memory and splits them up into one training, evaluation and
testing datasets with the sizes specifies below.

"""
n_training_features = 12000
n_evaluation_features = 2000
n_testing_features = 5000

rows = []
with open('../_data/steering_log_testing.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        rows.append(row)
    print(rows.__len__())
with open('../_data/steering_log_training.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        rows.append(row)
    print(rows.__len__())

random.seed(1532510560)  # this makes the random number generator produce reproducible results
random.shuffle(rows)
training_data = rows[0:12000]
evaluation_data = rows[12000:14000]
testing_data = rows[14000:19000]
with open('../_data/hetzell_training_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in training_data:
        spamwriter.writerow(row)
with open('../_data/hetzell_evaluation_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in evaluation_data:
        spamwriter.writerow(row)
with open('../_data/hetzell_testing_data.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in testing_data:
        spamwriter.writerow(row)

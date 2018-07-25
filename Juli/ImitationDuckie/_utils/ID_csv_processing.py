import csv
import random

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
training_data = rows[0:2000]
evaluation_data = rows[2000:2900]
testing_data = rows[2900:3900]
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

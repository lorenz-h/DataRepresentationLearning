import csv
import numpy as np


def load_labels():
    labels = []
    for csv_file in ["hetzell_raw_training_data.csv",
                     "hetzell_raw_testing_data.csv",
                     "hetzell_raw_evaluation_data.csv"]:
        with open('../_data/'+csv_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                labels.append(float(row[0]))
    return labels


def main():
    labels = load_labels()
    mean = np.mean(labels)
    mean_array = np.ones_like(labels)*mean

    delta = np.abs(mean_array-labels)

    static_error = np.mean(delta)
    print(static_error)


if __name__ == '__main__':
    main()

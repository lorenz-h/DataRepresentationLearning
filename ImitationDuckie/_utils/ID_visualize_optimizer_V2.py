import csv
import matplotlib.pyplot as plt
import numpy as np


def generate_2d_scatterplot(y, x, y_label, x_label, x_scale="linear", y_scale="linear"):
    dpi = 320
    plt.figure(figsize=(2000/dpi, 1500/dpi), dpi=dpi)
    g = np.arange(min(x), max(x), step=0.1)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def generate_4d_scatterplot(a, b, c, d):
    dpi = 320
    plt.figure(figsize=(2000 / dpi, 1500 / dpi), dpi=dpi)
    sizes = [item*50 for item in d]
    colors = c
    plt.scatter(a, b, s=sizes, c=colors, cmap='Greens')
    plt.show()


def main():
    learning_rates = []
    results = []
    n_convs = []
    n_dense = []
    with open("../_logs/optimizer_log.csv") as logfile:
        reader = csv.reader(logfile, delimiter=";", quotechar='"')
        for row in reader:
            results.append(float(row[0]))
            learning_rates.append(float(row[1]))
            n_convs.append(int(row[2]))
            n_dense.append(int(row[3]))
    generate_2d_scatterplot(results, learning_rates, "Loss", "Learning Rate", x_scale="log")
    generate_2d_scatterplot(results, n_convs, "Loss", "# of Convolutions")
    generate_4d_scatterplot(learning_rates, n_dense, n_convs, n_convs)


if __name__ == "__main__":
    main()

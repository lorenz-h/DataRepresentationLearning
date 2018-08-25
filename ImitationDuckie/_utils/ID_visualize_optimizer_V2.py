import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def generate_2d_scatterplot(y, x, y_label, x_label, x_scale="linear", y_scale="linear"):
    dpi = 320
    plt.figure(figsize=(2000/dpi, 1500/dpi), dpi=dpi)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.scatter(x, y)
    plt.ylim(ymax=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def generate_3d_scatterplot(y, x, y_label, x_label, color, x_scale="linear", y_scale="linear"):
    dpi = 320
    plt.figure(figsize=(2000/dpi, 1500/dpi), dpi=dpi)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.scatter(x, y, c=np.sqrt(color), cmap='Greens')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def generate_heatmap(x, y, z):
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    grid_z2 = griddata(np.array(list(zip(x,y))), z, (grid_x, grid_y), method='cubic')
    plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')


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
    generate_heatmap(n_convs, n_dense, results)
    generate_2d_scatterplot(results, learning_rates, "Loss", "Learning Rate", x_scale="log")
    generate_2d_scatterplot(results, n_convs, "Loss", "# of Convolutions")
    generate_2d_scatterplot(results, n_dense, "Loss", "# of Dense Nodes")
    generate_3d_scatterplot(n_convs, n_dense, "n_convs", "n_dense", results)
    generate_heatmap(n_convs, n_dense, results)


if __name__ == "__main__":
    main()

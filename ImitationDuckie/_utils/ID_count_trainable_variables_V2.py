"""
This script is used to count the number of trainable variables for the best networks from all experiments.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

saving = True
my_dpi = 350
scale = 0.8
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


cache_folder = "_cache_ID_results_visualization"


def conv_create_list_labeled_dataset(experiments):
    lldata = []
    for exp in experiments:
        csv_file = "../_saved_logs/_logs_" + exp + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [exp[0:3] for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        lldata.append(result)

    datasets = pd.concat(lldata, axis=0)
    return lldata


def dense_create_list_labeled_dataset(experiments):
    lldata = []
    for exp in experiments:
        csv_file = "../_saved_logs/_logs_" + exp + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [exp[0:3] for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "dense_nodes", "size_convergence", "n_layers"]
        lldata.append(result)

    datasets = pd.concat(lldata, axis=0)
    return lldata


def plotting(sizes, use_conv, maximus):
    normalizer = [size[0] for size in sizes if size[1][0:3] == maximus]
    assert len(normalizer) is not 0
    normalizer = np.mean(normalizer)

    fig, ax = plt.subplots(figsize=(2.5/scale, 2.5/scale), dpi=my_dpi)
    sns.barplot(x=[size[1] for size in sizes],
                y=[size[0]/normalizer for size in sizes],
                palette="cubehelix",
                errwidth=None)
    ax.set_ylabel("Relative size   " + r'$\frac{\#\theta_i}{\max(\#\theta)}$')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        if use_conv:
            fig.savefig(cache_folder + "/conv_size.png")
        else:
            fig.savefig(cache_folder + "/dense_size.png")


def calc_conv_sizes():
    experiments = ["RAW ConvNet", "DCT ConvNet", "SLT ConvNet"]
    lldata = conv_create_list_labeled_dataset(experiments)
    ###############################################################################
    sizes = []
    for dt in lldata:
        for index, row in dt.iterrows():
            n_convs = row["n_convs"]
            dense_nodes = row["dense_nodes"]
            depth_div = row["depth_div"]
            network_size = 0
            for i in range(int(n_convs)):
                network_size += (80*60)/(np.power(2, i)) * int(depth_div * (i + 1) * 16)
            network_size += 1.5*dense_nodes
            sizes.append([network_size, row["experiment"]])
    plotting(sizes, True, "RAW")


def calc_dense_sizes():
    experiments = ["RAW DenseNet", "DCT DenseNet", "SLT DenseNet"]
    lldata = dense_create_list_labeled_dataset(experiments)
    sizes = []
    for dt in lldata:
        for index, row in dt.iterrows():
            dense_nodes = row["dense_nodes"]
            size_convergence = row["size_convergence"]
            n_layers = row["n_layers"]
            network_size = 0
            for i in range(n_layers):
                network_size += int(dense_nodes/(i*size_convergence+1))
            sizes.append([network_size, row["experiment"]])
    plotting(sizes, False, "DCT")


def main():
    calc_conv_sizes()
    calc_dense_sizes()


if __name__ == '__main__':
    main()
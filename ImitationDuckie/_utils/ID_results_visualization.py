import csv

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def make_barplot(height, ylabel):
    bars = ('raw', 'dct', 'slt')
    y_pos = list(range(len(bars)))
    plt.bar(y_pos, height, width=0.6, color=sns.color_palette("cubehelix", 3), alpha=0.75)
    plt.xticks(y_pos, bars)
    plt.ylabel(ylabel)
    plt.show()


def check_overfitting():
    results = []
    for experiment in ["raw_cnn", "raw_dnn", "dct_cnn", "dct_dnn", "slt_cnn", "slt_dnn"]:
        testdata = []
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        testdata.append(result)

        testdata = pd.concat(testdata, axis=0)

        evaldata = []

        csv_file = "../_saved_logs/_logs_" + experiment + "/optimizer_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        evaldata.append(result)

        evaldata = pd.concat(evaldata, axis=0)

        print("++++++")
        keys = ["learning_rate", "n_convs", "dense_nodes", "depth_div", "experiment"]
        eval_losses = []
        for index, row in testdata.iterrows():
            thisdata = evaldata.copy()
            for key in keys:
                val = row[key]
                thisdata = thisdata.loc[evaldata[key] == val]
            print(thisdata.shape)
            eval_losses.append(float(thisdata["loss"]))
        eval_losses = np.asarray(eval_losses)
        testdata["eval_loss"] = eval_losses

        testdata['delta_loss'] = testdata["loss"]-testdata["eval_loss"]
        #  testdata['delta_loss'] = testdata['delta_loss'].abs()

        result = pd.concat([testdata['delta_loss'], testdata["experiment"]], axis=1)
        results.append(result)
    results = pd.concat(results, axis=0)
    print(results.shape)

    sns.catplot(x="experiment", y="delta_loss", data=results, jitter=True, alpha=0.3)
    sns.pointplot(x="experiment", y="delta_loss",
                  data=results, dodge=.532, join=False,
                  markers="d", scale=1.2, ci=None, palette="cubehelix")
    plt.ylabel("overfitting")
    plt.ylim((-0.01, 0.04))
    plt.show()


def testing_results():
    datasets = []
    for experiment in ["raw_cnn", "raw_dnn", "dct_cnn", "dct_dnn", "slt_cnn", "slt_dnn"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels

        result = pd.concat([df["labels"], df[0], df[1]], axis=1)
        result.columns = ["experiment", "loss", "learning rate"]
        datasets.append(result)

    datasets = pd.concat(datasets, axis=0)

    metric = (0.135-datasets["loss"])/0.135
    metric = np.asarray(metric)
    datasets["metric"] = metric

    print(list(datasets.columns.values))
    sns.catplot(x="experiment", y="loss", data=datasets, jitter=True, alpha=0.3)
    sns.pointplot(x="experiment", y="loss",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.2, ci=None, palette="cubehelix")
    plt.show()
    sns.catplot(x="experiment", y="metric", data=datasets, jitter=True, alpha=0.3)
    sns.pointplot(x="experiment", y="metric",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.2, ci=None, palette="cubehelix")
    plt.ylabel("relative performance")
    plt.show()


def optimizer_results():
    datasets = []
    for experiment in ["raw_cnn", "raw_dnn", "dct_cnn", "dct_dnn", "slt_cnn", "slt_dnn"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/optimizer_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels

        result = pd.concat([df["labels"], df[0], df[1]], axis=1)
        result.columns = ["experiment", "loss", "learning rate"]
        datasets.append(result)

    datasets = pd.concat(datasets, axis=0)

    metric = (0.135-datasets["loss"])/0.135
    metric = np.asarray(metric)
    datasets["metric"] = metric

    print(list(datasets.columns.values))
    sns.catplot(x="experiment", y="loss", data=datasets, jitter=True, alpha=0.3)
    sns.pointplot(x="experiment", y="loss",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.2, ci=None, palette="cubehelix")
    plt.show()
    sns.catplot(x="experiment", y="metric", data=datasets, jitter=True, alpha=0.3)
    sns.pointplot(x="experiment", y="metric",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.2, ci=None, palette="cubehelix")
    plt.ylabel("relative performance")
    plt.show()


def conv_testing_results():
    lldata = []
    for experiment in ["raw_cnn", "dct_cnn", "slt_cnn"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        lldata.append(result)

    datasets = pd.concat(lldata, axis=0)
    print(list(datasets.columns.values))

    ###############################################################################
    heights = []
    for dt in lldata:
        params = 0
        n_convs = dt["n_convs"].mean()
        depth_divergence = dt["depth_div"].mean()
        for i in range(int(n_convs)):
            params += ((80*60)/(2 ^ i))*depth_divergence*(i+1)*16
        heights.append(params)

    make_barplot(heights, ylabel="conv network size")


def dense_testing_results():
    lldata = []
    for experiment in ["raw_dnn", "dct_dnn", "slt_dnn"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "dense_nodes", "size_convergence", "n_layers"]
        lldata.append(result)

    datasets = pd.concat(lldata, axis=0)
    print(list(datasets.columns.values))
    # Make fake dataset
    """
    height = [dt["dense_nodes"].mean() for dt in lldata]
    make_barplot(height, ylabel="dense nodes")

    height = [dt["n_layers"].mean() for dt in lldata]
    make_barplot(height, ylabel="# layers")

    height = [dt["size_convergence"].mean() for dt in lldata]
    make_barplot(height, ylabel="size convergence")
    """

    complexities = []
    for dt in lldata:
        n_layers = dt["n_layers"].mean()
        dense_nodes = dt["dense_nodes"].mean()
        size_convergence = dt["size_convergence"].mean()
        complexity = (dense_nodes*n_layers)-(n_layers*(dense_nodes-(dense_nodes/(((n_layers-1)*size_convergence)+1)))/2)
        complexities.append(complexity)
    make_barplot(complexities, ylabel="dense network size")


if __name__ == "__main__":
    sns.set(palette="cubehelix", style="white")
    check_overfitting()

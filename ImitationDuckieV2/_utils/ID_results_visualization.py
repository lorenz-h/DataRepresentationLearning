import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

my_dpi = 100
saving = True

cache_folder = "_cache_ID_results_visualization"


def make_barplot(height, ylabel):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=my_dpi)
    bars = ('RAW', 'DCT', 'SLT')
    y_pos = list(range(len(bars)))
    plt.bar(y_pos, height, width=0.4, color=sns.color_palette("cubehelix", 3), alpha=0.5)
    ax.set_xticks(np.arange(3), bars)
    ax.set_ylabel(ylabel)
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder+"/" + ylabel + ".png")


def remove_outlier(df, name):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df


def check_overfitting():
    results = []
    for experiment in ["RAW ConvNet", "RAW DenseNet", "DCT ConvNet", "DCT DenseNet", "SLT ConvNet", "SLT DenseNet"]:
        testdata = []
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        result = remove_outlier(result, "loss")
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

        result = pd.concat([testdata['delta_loss'], testdata["experiment"]], axis=1)
        results.append(result)
    results = pd.concat(results, axis=0)
    print(results.shape)

    ##
    ##
    ##
    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.catplot(ax=ax, x="experiment", y="delta_loss", data=results, jitter=True, alpha=0.3, palette="cubehelix")
    sns.pointplot(ax=ax, x="experiment", y="delta_loss",
                  data=results, dodge=.532, join=False,
                  markers="d", scale=1.6, ci=None, palette="cubehelix")
    ax.set_ylabel("loss delta")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.set_ylim((-0.01, 0.04))
    fig.show()
    if saving:
        fig.savefig(cache_folder+"/overfitting.png")
    ##
    ##
    ##


def tt_results():
    datasets = []
    for experiment in ["Raw ConvNet", "Raw DenseNet", "DCT ConvNet", "DCT DenseNet", "SLT ConvNet", "SLT DenseNet"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")
        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels

        result = pd.concat([df["labels"], df[0], df[1]], axis=1)
        result.columns = ["experiment", "loss", "learning rate"]
        result = remove_outlier(result, "loss")
        datasets.append(result)

    datasets = pd.concat(datasets, axis=0)

    metric = 1 - ((0.135-datasets["loss"])/0.135)
    metric = np.asarray(metric)
    datasets["metric"] = metric

    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.catplot(ax=ax, x="experiment", y="loss", data=datasets, jitter=True, alpha=0.3, palette="cubehelix")
    plt.ylabel("absolute loss")
    sns.pointplot(ax=ax, x="experiment", y="loss",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.6, ci=None, palette="cubehelix")
    ax.set_ylabel("absolute loss")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder+"/absolute_loss.png")
    ##
    ##
   
    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.catplot(ax=ax, x="experiment", y="metric", data=datasets, jitter=True, alpha=0.3, palette="cubehelix")
    plt.ylabel("relative loss")
    sns.pointplot(ax=ax, x="experiment", y="metric",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.6, ci=None, palette="cubehelix")
    ax.set_ylabel("relative loss")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder+"/relative_loss.png")
    ##
    ##
    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.violinplot(ax=ax, x="experiment", y="loss", data=datasets,  palette="cubehelix")
    plt.ylabel("absolute loss")
    ax.set_ylabel("absolute loss")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder + "/absolute_loss_violin.png")
    ##
    ##

    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.violinplot(ax=ax, x="experiment", y="metric", data=datasets, palette="cubehelix")
    plt.ylabel("relative loss")
    ax.set_ylabel("relative loss")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder + "/relative_loss_violin.png")


def conv_testing_results():
    lldata = []
    for experiment in ["RAW ConvNet", "DCT ConvNet", "SLT ConvNet"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "n_convs", "dense_nodes", "depth_div"]
        result = remove_outlier(result, "loss")
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

    fig, ax = plt.subplots(figsize=(5, 5), dpi=my_dpi)
    sns.barplot(x=["RAW", "DCT", "SLT"], y=heights, palette="cubehelix", errwidth=None)
    fig.show()
    if saving:
        fig.savefig(cache_folder+"/conv_size.png")


def dense_testing_results():
    lldata = []
    for experiment in ["RAW DenseNet", "DCT DenseNet", "SLT DenseNet"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")

        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels
        result = pd.concat([df["labels"], df[0], df[1], df[2], df[3], df[4]], axis=1)
        result.columns = ["experiment", "loss", "learning_rate", "dense_nodes", "size_convergence", "n_layers"]
        result = remove_outlier(result, "loss")
        lldata.append(result)

    complexities = []
    for dt in lldata:
        n_layers = dt["n_layers"].mean()
        dense_nodes = dt["dense_nodes"].mean()
        size_convergence = dt["size_convergence"].mean()
        complexity = (dense_nodes*n_layers)-(n_layers*(dense_nodes-(dense_nodes/(((n_layers-1)*size_convergence)+1)))/2)
        complexities.append(complexity)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=my_dpi)
    sns.barplot(x=["RAW", "DCT", "SLT"], y=complexities, palette="cubehelix", errwidth=None)
    fig.show()
    if saving:
        fig.savefig(cache_folder + "/dense_size.png")


def optimizer_results():
    datasets = []
    for experiment in ["Raw ConvNet", "Raw DenseNet", "DCT ConvNet", "DCT DenseNet", "SLT ConvNet", "SLT DenseNet"]:
        csv_file = "../_saved_logs/_logs_" + experiment + "/optimizer_results.csv"
        df = pd.read_csv(csv_file, header=None, delimiter=";")
        labels = [experiment for _ in range(int(df[0].size))]
        labels = np.asarray(labels)
        df["labels"] = labels

        result = pd.concat([df["labels"], df[0], df[1]], axis=1)
        result.columns = ["experiment", "loss", "learning rate"]
        result = remove_outlier(result, "loss")
        datasets.append(result)

    datasets = pd.concat(datasets, axis=0)

    metric = 1 - ((0.135 - datasets["loss"]) / 0.135)
    metric = np.asarray(metric)
    datasets["metric"] = metric

    fig, ax = plt.subplots(figsize=(8, 5), dpi=my_dpi)

    sns.catplot(ax=ax, x="experiment", y="loss", data=datasets, jitter=True, alpha=0.3, palette="cubehelix")
    plt.ylabel("absolute loss")
    sns.pointplot(ax=ax, x="experiment", y="loss",
                  data=datasets, dodge=.532, join=False,
                  markers="d", scale=1.6, ci=None, palette="cubehelix")
    ax.set_ylabel("absolute loss")
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    fig.show()
    if saving:
        fig.savefig(cache_folder + "/optimizer_absolute_loss.png")


def main():
    sns.set(palette="cubehelix", style="whitegrid")
    tt_results()
    check_overfitting()
    conv_testing_results()
    dense_testing_results()
    optimizer_results()


if __name__ == "__main__":
    main()


import pandas as pd
import numpy as np


def get_dataset(experiment):
    csv_file = "../_saved_logs/_logs_" + experiment + "/testing_results.csv"
    df = pd.read_csv(csv_file, header=None, delimiter=";")
    labels = [experiment for _ in range(int(df[0].size))]
    labels = np.asarray(labels)
    df["labels"] = labels
    result = pd.concat([df["labels"], df[0], df[1]], axis=1)
    result.columns = ["experiment", "loss", "learning rate"]
    return result


def main():
    for experiment in ["RAW ConvNet", "RAW DenseNet", "DCT ConvNet", "DCT DenseNet", "SLT ConvNet", "SLT DenseNet"]:
        df = get_dataset(experiment)
        print(df["experiment"][0])
        print(round(df["loss"].mean(),5))
        print(round(df["loss"].std(),6))


if __name__ == '__main__':
    main()
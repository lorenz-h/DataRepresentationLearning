import pickle

filepath = "Dataset/Training/sample690.pkl"

pkl_file = open(filepath, 'rb')

data1 = pickle.load(pkl_file)
print(data1[1])
import skopt.plots
import pickle

results = pickle.load(open("../_logs/optimizer_points.pkl", "rb"))
skopt.plots.plot_regret(results)

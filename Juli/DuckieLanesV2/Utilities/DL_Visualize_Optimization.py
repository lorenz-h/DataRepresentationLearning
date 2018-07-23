import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


setups = pickle.load(open("../logs/optimizer_points_shearlet.pkl", "rb"))
results, points = zip(*setups)
learning_rate, n_epochs, n_convolutions, n_dense_nodes = zip(*points)
results = np.clip(results, 0.0, 1.0)

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter3D(learning_rate, n_epochs, zs=n_convolutions, c=results, cmap='summer')
ax.set_xlabel('learning_rate')
ax.set_ylabel('n_epochs')
ax.set_zlabel('n_convolutions')

cbar = fig.colorbar(p, ax=ax)
cbar.set_label('absolute average error', rotation=90)
plt.show()

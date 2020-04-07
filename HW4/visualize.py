import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

from data import *

DIR = './figures/target_curves/'

data = get_data()

for key in data.keys():
    run = get_all_data(data, key)
    target = get_target_data(run)
    mat = np.transpose(target.to_numpy(dtype=float)[:, 1:])
    print(mat.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z = mat[2]
    x = mat[0]
    y = mat[1]
    ax.plot(x, y, z, label='trajectory')
    ax.legend()
    plt.savefig(DIR + key + '_curve.png')

    plt.close()

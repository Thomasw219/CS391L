import matplotlib.pyplot as plt
import numpy as np

from gp import *
from data import *

X_SCALE = 10
Y_SCALE = 10
STEP = 0.05
T_I = 0
T_F = 17.5
INIT_SIG_F = 2.33219503
INIT_SIG_L = -4.55283543
INIT_SIG_N = -1.59004312
INTERVAL = 2
time_centers = np.arange(T_I, T_F, STEP)

def filter_array(array, lower, upper):
    l = 0
    u = 0
    for i, val in enumerate(array):
        if val >= lower:
            l = i
            break
    for i, val in enumerate(array[l:]):
        if upper < val:
            break
        u = i + l
    return array[l:u + 1]

means = np.load("means.npy")
sig_f = np.load("sig_f.npy")
sig_l = np.load("sig_l.npy")
sig_n = np.load("sig_n.npy")

KEY = 'AG'
MARKER = 15
DIM = 'x'
DIR = './figures/graphs'

data = get_data()
run = get_all_data(data, KEY)
marker_data = get_marker_data(run, MARKER, DIM)

bounds = [T_I, T_F]
init_sig_f = [INIT_SIG_F, INIT_SIG_F]
init_sig_l = [INIT_SIG_L, INIT_SIG_L]
init_sig_n = [INIT_SIG_N, INIT_SIG_N]

local_means = {}
for t in time_centers:
    local_means[t * X_SCALE] = []

for i, t in enumerate(time_centers):
    print(t)
    filtered_marker_data = restrict_times(marker_data, t - INTERVAL / 2, t + INTERVAL / 2)
    X, Y = to_matrices(filtered_marker_data)
    N = X.shape[0]

    indices = np.random.choice(N, size=N, replace=False)
    X = X[indices] * X_SCALE
    Y = Y[indices] * Y_SCALE
    m_y = np.ones((N, 1)) * np.mean(Y)

    subset = np.array(filter_array(time_centers, t - 0.025, t + 0.025)) * X_SCALE
    m_f = np.ones((subset.shape[0], 1)) * np.mean(Y)
    sample_mean, _ = define_GP(X, Y, m_y, m_f, subset, sig_f[i], sig_l[i], sig_n[i])

    print(sample_mean)
    for i in range(sample_mean.shape[0]):
        local_means[np.asscalar(subset[i])].append(sample_mean[i])

averaged_means = [np.mean(local_means[t * X_SCALE]) for t in time_centers]

plt.figure(0)
plt.plot(time_centers, means, c='r', label='Global Params Mean')
plt.plot(time_centers, averaged_means, c='b', label='Local Params Mean')
plt.legend()
plt.savefig(DIR + '/mean_comparison.png')


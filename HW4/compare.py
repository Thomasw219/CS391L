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

KEY = 'AG'
MARKER = 15
DIM = 'x'
DIR = './figures/' + KEY + '_' + str(INTERVAL) + '_comparison'

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

def nearest_center_index(centers, x):
    if centers.shape[0] == 1:
        return 0
    middle = (centers.shape[0] - 1) // 2
    upper_middle = np.asscalar(centers[middle + 1])
    lower_middle = np.asscalar(centers[middle])
    is_lower = lower_middle - x > 0
    upper_closer = np.abs(upper_middle - x) <= np.abs(lower_middle- x)
    choose_upper = np.abs(upper_middle - x) <= STEP / 2
    choose_lower = np.abs(lower_middle - x) <= STEP / 2
    if choose_upper:
        return middle + 1 + nearest_center_index(centers[middle + 1:], x)
    elif choose_lower or is_lower:
        return nearest_center_index(centers[:middle + 1], x)
    else:
        return middle + 1 + nearest_center_index(centers[middle + 1:], x)

def square_error(y1, y2):
    return (np.asscalar(y1) - np.asscalar(y2))**2

data = get_data()

sig_f = np.load("./sig_f.npy")
sig_l = np.load("./sig_l.npy")
sig_n = np.load("./sig_n.npy")

data_3_train, data_3_test = get_all_data(data, KEY, red=3)
third_data = get_marker_data(data_3_train, MARKER, DIM)
third_test = get_marker_data(data_3_test, MARKER, DIM)
data_10_train, data_10_test = get_all_data(data, KEY, red=10)
tenth_data = get_marker_data(data_10_train, MARKER, DIM)
tenth_test = get_marker_data(data_10_test, MARKER, DIM)
X_10, Y_10 = to_matrices(tenth_data)
X_10 = X_10 * X_SCALE
Y_10 = Y_10 * Y_SCALE
X_10_test, Y_10_test = to_matrices(tenth_test)
X_10_test = X_10_test * X_SCALE
Y_10_test = Y_10_test * Y_SCALE
X_3_test, Y_3_test = to_matrices(third_test)
Y_3_test = Y_3_test * Y_SCALE
m_y_10 = np.ones(Y_10.shape) * np.mean(Y_10)
print(X_10_test.shape)
print(X_3_test.shape)

global_sum = 0
local_sum = 0

m_f = np.ones((X_10_test.shape[0], 1)) * np.mean(Y_10)
global_mean, _ = define_GP(X_10, Y_10, m_y_10, m_f, X_10_test, INIT_SIG_F, INIT_SIG_L, INIT_SIG_N)
for i, x in enumerate(X_10_test):
    global_sum += square_error(global_mean[i], Y_10_test[i])
    print("Global MSE: {}".format(global_sum / (i + 1)))


for i, x in enumerate(X_3_test):
    c_i = nearest_center_index(time_centers, np.asscalar(x))
    t = time_centers[c_i]
    filtered_marker_data = restrict_times(third_data, t - INTERVAL / 2, t + INTERVAL / 2)
    X_3, Y_3 = to_matrices(filtered_marker_data)

    X_3 = X_3 * X_SCALE
    Y_3 = Y_3 * Y_SCALE
    N = X_3.shape[0]

    y = Y_3_test[i]
    m_y = np.ones((N, 1)) * np.mean(Y_3)

#    subset = np.array(filter_array(time_centers, t - INTERVAL / 2, t + INTERVAL / 2))
#    subset = np.array(filter_array(time_centers, t - .025, t + .025))
#    subset = np.reshape(subset, (subset.shape[0], 1)) * X_SCALE
    subset = np.reshape(np.array([x * X_SCALE]), (1, 1))
    m_f = np.ones((subset.shape[0], 1)) * np.mean(Y_3)
    local_mean, _ = define_GP(X_3, Y_3, m_y, m_f, subset, sig_f[c_i], sig_l[c_i], sig_n[c_i])


    local_sum += square_error(local_mean, y)
    print("x: {}\t center: {}".format(x, t))
    print("Local MSE: {}".format(local_sum / (i + 1)))



exit()

means = np.load("./means.npy")

bounds = [T_I, T_F]
init_sig_f = [INIT_SIG_F, INIT_SIG_F]
init_sig_l = [INIT_SIG_L, INIT_SIG_L]
init_sig_n = [INIT_SIG_N, INIT_SIG_N]

plt.figure(0)
plt.plot(time_centers, means, c='r', label='Global Params Mean')
plt.plot(time_centers, averaged_means, c='b', label='Local Params Mean')
plt.legend()
plt.savefig(DIR + '/' + KEY + '_' + str(MARKER) + '_' + DIM + '_mean_comparison.png')
"""
plt.plot(time_centers, means, c='k', label='trajectory')
plt.plot(time_centers, sig_f, c='r', label='sig_f')
plt.plot(time_centers, sig_l, c='g', label='sig_l')
plt.plot(time_centers, sig_n, c='b', label='sig_n')
plt.plot(bounds, init_sig_f, c='#ffcccb', label='initial sig_f')
plt.plot(bounds, init_sig_l, c='#90ee90', label='initial sig_l')
plt.plot(bounds, init_sig_n, c='#add8e6', label='initial sig_n')
plt.legend()
plt.savefig('./figures/AG_15_X_hyperparams.png')
"""

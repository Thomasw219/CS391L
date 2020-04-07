import numpy as np

from scipy.optimize import minimize

from gp import *
from data import *

def callback(xk):
    print(xk)

DIR = './figures/optimize_from_last_params'

X_SCALE = 10
Y_SCALE = 10
T_I = 0
T_F = 17.5

STEP = 0.05
N_SAMPLES = 200
INIT_SIG_F = 2.33219503
INIT_SIG_L = -4.55283543
INIT_SIG_N = -1.59004312
INTERVAL = 2
time_centers = np.arange(T_I, T_F, STEP)
"""
N_SAMPLES = 500
INIT_SIG_F = 0
INIT_SIG_L = 0
INIT_SIG_N = 0
INTERVAL = 17.5
time_centers = [8.75]
"""

init_sig = np.array([INIT_SIG_F, INIT_SIG_L, INIT_SIG_N])

data = get_data()
run = get_all_data(data, 'AG')
marker_data = get_marker_data(run, 15, 'x')

sig_f = []
sig_l = []
sig_n = []

for t in time_centers:
    filtered_marker_data = restrict_times(marker_data, t - INTERVAL / 2, t + INTERVAL / 2)
    X, Y = to_matrices(filtered_marker_data)
    N = X.shape[0]
    print((t - INTERVAL / 2, t + INTERVAL / 2))
    print(N)
#    indices = np.random.choice(N, size=min(N, N_SAMPLES), replace=False)
    indices = np.random.choice(N, size=N, replace=False)
    X = X[indices] * X_SCALE
    Y = Y[indices] * Y_SCALE
#    N = min(N, N_SAMPLES)

    m_y = np.ones((N, 1)) * np.mean(Y)

    N = X.shape[0]

    res = minimize(neglogprob, init_sig, args=(X, Y - m_y, N), method='BFGS', jac=neglogprob_grad, options={'disp': True})
    print(res.x)

    sig_f.append(res.x[0])
    sig_l.append(res.x[1])
    sig_n.append(res.x[2])
    init_sig = np.ravel(res.x)

    sample_X = np.arange((t - INTERVAL / 2) * X_SCALE, (t + INTERVAL / 2) * X_SCALE, INTERVAL * X_SCALE / N_SAMPLES)
    sample_X = np.reshape(sample_X, (sample_X.shape[0], 1))
    m_f = np.ones((sample_X.shape[0], 1)) * np.mean(Y)

    sample_mean, sample_cov_mat = define_GP(X, Y, m_y, m_f, sample_X, sig_f[-1], sig_l[-1], sig_n[-1])
    sample_mean = np.ravel(sample_mean)
    sample_var = np.array([sample_cov_mat[i][i] for i in range(sample_cov_mat.shape[0])])

    plt.figure(0)
    plt.plot(sample_X, sample_mean, color='r')
    plt.plot(sample_X, sample_mean + 2 * sample_var, color='b')
    plt.plot(sample_X, sample_mean - 2 * sample_var, color='b')
    plt.scatter(np.ravel(X), np.ravel(Y), c='g')
    plt.savefig(DIR + '/AG_15_X_center_' + str(t) + '_interval_' + str(INTERVAL) + '.png')
    plt.close()

print(sig_f)
print(sig_l)
print(sig_n)
bounds = [T_I, T_F]
init_sig_f = [INIT_SIG_F, INIT_SIG_F]
init_sig_l = [INIT_SIG_L, INIT_SIG_L]
init_sig_n = [INIT_SIG_N, INIT_SIG_N]

plt.figure(0)
plt.plot(time_centers, sig_f, c='r', label='sig_f')
plt.plot(time_centers, sig_l, c='g', label='sig_l')
plt.plot(time_centers, sig_n, c='b', label='sig_n')
plt.plot(bounds, init_sig_f, c='#ffcccb', label='initial sig_f')
plt.plot(bounds, init_sig_l, c='#90ee90', label='initial sig_l')
plt.plot(bounds, init_sig_n, c='#add8e6', label='initial sig_n')
plt.legend()
plt.savefig(DIR + '/AG_15_X_hyperparams.png')

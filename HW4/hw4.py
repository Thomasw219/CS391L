import numpy as np

from scipy.optimize import minimize

from gp import *
from data import *

def callback(xk):
    print(xk)

T_I = 0
T_F = 18
X_SCALE = 10
Y_SCALE = 10
N_SAMPLES = 100

data = get_data()
run = get_all_data(data, 'AG')
marker_data = get_marker_data(run, 15, 'x')
marker_data = restrict_times(marker_data, T_I, T_F)
X, Y = to_matrices(marker_data)
N = X.shape[0]
indices = np.random.choice(N, size=300, replace=False)
X = X[indices] * X_SCALE
Y = Y[indices] * Y_SCALE

INIT_SIG_F = 0
INIT_SIG_L = 0
INIT_SIG_N = 0
INIT_SIG = np.array([INIT_SIG_F, INIT_SIG_L, INIT_SIG_N])

N = X.shape[0]

res = minimize(neglogprob, INIT_SIG, args=(X, Y, N), method='BFGS', jac=neglogprob_grad, options={'disp': True}, callback=callback)
print(res.x)

SIG_F = res.x[0]
SIG_L = res.x[1]
SIG_N = res.x[2]

m_y = np.ones((N, 1)) * np.mean(Y)
sample_X = np.arange(T_I * X_SCALE, T_F * X_SCALE, (T_F - T_I) * X_SCALE / N_SAMPLES)
sample_X = np.reshape(sample_X, (sample_X.shape[0], 1))
m_f = np.ones((sample_X.shape[0], 1)) * np.mean(Y)
print(np.mean(Y))

sample_mean, sample_cov_mat = define_GP(X, Y, m_y, m_f, sample_X, SIG_F, SIG_L, SIG_N)
sample_mean = np.ravel(sample_mean)
sample_var = np.array([sample_cov_mat[i][i] for i in range(sample_cov_mat.shape[0])])

print(sample_mean)
print(sample_var)
plt.figure(0)
plt.plot(sample_X, sample_mean, color='r')
plt.plot(sample_X, sample_mean + 2 * sample_var, color='b')
plt.plot(sample_X, sample_mean - 2 * sample_var, color='b')
plt.scatter(np.ravel(X), np.ravel(Y), c='g')
plt.savefig('./figures/graph.png')

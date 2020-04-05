import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

def delta(x_1, x_2):
    if np.array_equal(x_1, x_2):
        return 1
    else:
        return 0

def kprime(x_1, x_2, sig_f, sig_l, sig_n):
    arg = -1 / 2 * np.exp(sig_l) * np.linalg.norm(x_1 - x_2)**2
    return np.exp(sig_f) * np.exp(arg)

def k(x_1, x_2, sig_f, sig_l, sig_n):
    return kprime(x_1, x_2, sig_f, sig_l, sig_n) + np.exp(sig_n) * delta(x_1, x_2)

def conditional_mean(K_yy_inv, K_fy, y, m_y, m_f):
    transform = np.matmul(K_fy, K_yy_inv)
    return np.matmul(transform, y - m_y) + m_f

def conditional_variance(K_ff, K_fy, K_yy_inv):
    temp = np.matmul(K_yy_inv, np.transpose(K_fy))
    return K_ff - np.matmul(K_fy, temp)

def Cov(X_1, X_2, ker, sig_f, sig_l, sig_n):
    K = np.zeros((X_1.shape[0], X_2.shape[0]))
    sym = np.array_equal(X_1, X_2)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if not sym:
                K[i, j] = ker(X_1[i], X_2[j], sig_f, sig_l, sig_n)
            else:
                if K[j, i] != 0:
                    K[i, j] = K[j, i]
                else:
                    K[i, j] = ker(X_1[i], X_2[j], sig_f, sig_l, sig_n)
    return K

def define_GP(data_X, data_Y, m_y, m_f, sample_X, sig_f, sig_l, sig_n):
    K_yy = Cov(data_X, data_X, k, sig_f, sig_l, sig_n)
    K_yy_inv = np.linalg.inv(K_yy)
    K_fy = Cov(sample_X, data_X, kprime, sig_f, sig_l, sig_n)
    K_ff = Cov(sample_X, sample_X, kprime, sig_f, sig_l, sig_n)
    sample_mean = conditional_mean(K_yy_inv, K_fy, data_Y, m_y, m_f)
    sample_cov_mat = conditional_variance(K_ff, K_fy, K_yy_inv)
    return sample_mean, sample_cov_mat

def dkdsig_f(x_1, x_2, sig_f, sig_l, sig_n):
    return kprime(x_1, x_2, sig_f, sig_l, sig_n)

def dkdsig_l(x_1, x_2, sig_f, sig_l, sig_n):
    arg = -1 / 2 * np.exp(sig_l) * np.linalg.norm(x_1 - x_2)**2
    return kprime(x_1, x_2, sig_f, sig_l, sig_n) * arg

def dkdsig_n(x_1, x_2, sig_f, sig_l, sig_n):
    return np.exp(sig_n) * delta(x_1, x_2)

def logprob(sig, *args):
    data_X = args[0]
    data_Y = args[1]

    sig_f = sig[0]
    sig_l = sig[1]
    sig_n = sig[2]

    Q = Cov(data_X, data_X, k, sig_f, sig_l, sig_n)
    Q_inv = np.linalg.inv(Q)
    term1 = -1 / 2 * np.matmul(np.matmul(np.transpose(data_Y), Q_inv), data_Y)
    term2 = -1 / 2 * np.log(np.linalg.det(Q))
    term3 = N / 2 * np.log(2 * np.pi)
    return term1 + term2 + term3

def neglogprob(sig, *args):
    return -1 * logprob(sig, *args)

def partial_derivative(Q_inv, dQdsig, data_Y):
    dQdsig_inv = -1 * np.matmul(np.matmul(Q_inv, dQdsig), Q_inv)
    dlogdetQdsig = np.trace(np.matmul(Q_inv, dQdsig))
    term1 = -1 / 2 * np.matmul(np.matmul(np.transpose(data_Y), dQdsig_inv), data_Y)
    term2 = -1 / 2 * dlogdetQdsig
    return term1 + term2

def logprob_grad(sig, *args):
    data_X = args[0]
    data_Y = args[1]

    sig_f = sig[0]
    sig_l = sig[1]
    sig_n = sig[2]

    Q = Cov(data_X, data_X, k, sig_f, sig_l, sig_n)
    Q_inv = np.linalg.inv(Q)
    dQdsig_f = Cov(data_X, data_X, dkdsig_f, sig_f, sig_l, sig_n)
    dQdsig_l = Cov(data_X, data_X, dkdsig_l, sig_f, sig_l, sig_n)
    dQdsig_n = Cov(data_X, data_X, dkdsig_n, sig_f, sig_l, sig_n)
    grad = np.zeros((3,))
    grad[0] = partial_derivative(Q_inv, dQdsig_f, data_Y)
    grad[1] = partial_derivative(Q_inv, dQdsig_l, data_Y)
    grad[2] = partial_derivative(Q_inv, dQdsig_n, data_Y)
    return grad

def neglogprob_grad(sig, *args):
    return -1 * logprob_grad(sig, *args)

INIT_SIG_F = 2.3
INIT_SIG_L = -7.8
INIT_SIG_N = 0
INIT_SIG = np.array([INIT_SIG_F, INIT_SIG_L, INIT_SIG_N])

N = 4
data_X = np.reshape(np.array([204, 90, 150, 400]), (N, 1))
data_Y = np.reshape(np.array([-7, -18, -10, 20]), (N, 1))

res = minimize(neglogprob, INIT_SIG, args=(data_X, data_Y), method='BFGS', jac=neglogprob_grad, options={'disp': True})
print(res.x)

SIG_F = res.x[0]
SIG_L = res.x[1]
SIG_N = res.x[2]

m_y = np.zeros((N,1))
sample_X = np.arange(0, 500, 1)
sample_X = np.reshape(sample_X, (sample_X.shape[0], 1))
m_f = np.zeros((sample_X.shape[0], 1))

sample_mean, sample_cov_mat = define_GP(data_X, data_Y, m_y, m_f, sample_X, SIG_F, SIG_L, SIG_N)
sample_mean = np.ravel(sample_mean)
sample_var = np.array([sample_cov_mat[i][i] for i in range(sample_cov_mat.shape[0])])

plt.figure(0)
plt.plot(sample_X, sample_mean, color='r')
plt.plot(sample_X, sample_mean + 2 * sample_var, color='b')
plt.plot(sample_X, sample_mean - 2 * sample_var, color='b')
plt.scatter(np.ravel(data_X), np.ravel(data_Y), c='g')
plt.show()

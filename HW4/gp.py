import numpy as np
import matplotlib.pyplot as plt

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

data_X = np.reshape(np.array([204, 90, 400]), (3, 1))
data_Y = np.array([-7, -18, 20])
N = data_X.shape[0]
def logprob(sig):
    sig_f = sig[0]
    sig_l = sig[1]
    sig_n = sig[2]

    Q = Cov(data_X, data_X, k, sig_f, sig_l, sig_n)
    term1 = -1 / 2 * np.matmul(np.matmul(np.transpose(data_Y), np.linalg.inv(Q)), data_Y)
    term2 = -1 / 2 * np.log(np.linalg.det(Q))
    term3 = N / 2 * log(2 * np.pi)
    return term1 + term2 + term3

SIG_F = 2.3
SIG_L = -7.8
SIG_N = 0

m_y = np.array([0, 0, 0])
sample_X = np.arange(0, 500, 1)
sample_X = np.reshape(sample_X, (sample_X.shape[0], 1))
m_f = np.zeros((sample_X.shape[0],))

sample_mean, sample_cov_mat = define_GP(data_X, data_Y, m_y, m_f, sample_X, SIG_F, SIG_L, SIG_N)
sample_var = np.array([sample_cov_mat[i][i] for i in range(sample_cov_mat.shape[0])])

print(sample_mean)
print(sample_var)
plt.figure(0)
plt.plot(sample_X, sample_mean, color='r')
plt.plot(sample_X, sample_mean + 2 * sample_var, color='b')
plt.plot(sample_X, sample_mean - 2 * sample_var, color='b')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def SE_kernel(x_1, x_2, sigma_f=10, sigma_l=1):
    arg = -1 * np.linalg.norm(x_1 - x_2)**2 / (2 * sigma_l**2)
    return sigma_f * np.exp(arg)

def delta_kernel(x_1, x_2, sigma_n=1):
    if np.array_equal(x_1, x_2):
        return sigma_n**2
    else:
        return 0

def conditional_mean(K_yy_inv, K_fy, y, m_y, m_f):
    transform = np.matmul(K_fy, K_yy_inv)
    return np.matmul(transform, y - m_y) + m_f

def conditional_variance(K_ff, K_fy, K_yy_inv):
    temp = np.matmul(K_yy_inv, np.transpose(K_fy))
    return K_ff - np.matmul(K_fy, temp)

def define_GP(data_X, data_Y, m_y, m_f, sample_X, sig_f, sig_l, sig_n):
    K_yy = np.zeros((data_X.shape[0], data_X.shape[0]))
    for i in range(data_X.shape[0]):
        for j in range(data_X.shape[0]):
            K_yy[i, j] = SE_kernel(data_X[i], data_X[j], sigma_f=SIG_F, sigma_l=SIG_L) + delta_kernel(data_X[i], data_X[j], sigma_n=SIG_N)
    K_yy_inv = np.linalg.inv(K_yy)

    K_fy = np.zeros((sample_X.shape[0], data_X.shape[0]))
    for i in range(sample_X.shape[0]):
        for j in range(data_X.shape[0]):
            K_fy[i, j] = SE_kernel(sample_X[i], data_X[j], sigma_f=SIG_F, sigma_l=SIG_L)

    K_ff = np.zeros((sample_X.shape[0], sample_X.shape[0]))
    for i in range(sample_X.shape[0]):
        for j in range(sample_X.shape[0]):
            K_ff[i, j] = SE_kernel(sample_X[i], sample_X[j], sigma_f=SIG_F, sigma_l=SIG_L)

    sample_mean = conditional_mean(K_yy_inv, K_fy, data_Y, m_y, m_f)
    sample_cov_mat = conditional_variance(K_ff, K_fy, K_yy_inv)
    return sample_mean, sample_cov_mat

SIG_F = 10
SIG_L = 75
SIG_N = 1

data_Y = np.array([-7, -18, 20])
data_X = np.reshape(np.array([204, 90, 400]), (3, 1))
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


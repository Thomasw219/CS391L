import numpy as np
import matplotlib.pyplot as plt

def SE_kernel(x_1, x_2, sigma_f=10, sigma_l=1):
    arg = -1 * np.linalg.norm(x_1 - x_2)**2 / (2 * sigma_l**2) 
    return sigma_f * np.exp(arg)

def conditional_mean(K_yy_inv, K_fy, y, m_y, m_f):
    transform = np.matmul(K_fy, K_yy_inv)
    return np.matmul(transform, y - m_y) + m_f

def conditional_variance(K_ff, K_fy, K_yy_inv):
    temp = np.matmul(K_yy_inv, np.transpose(K_fy))
    return K_ff - np.matmul(K_fy, temp)

SIG_F = 10
SIG_L = 50
SIG_N = 1

Y = np.array([-7, -18])
data_x = np.array([204, 90])
m_y = np.array([0, 0])
m_f = np.array([0])
sample_X = np.arange(0, 500, 1)
sample_mean = np.zeros(sample_X.shape)
sample_var = np.zeros(sample_X.shape)

K_yy = np.zeros((data_x.shape[0], data_x.shape[0]))
for i in range(data_x.shape[0]):
    for j in range(data_x.shape[0]):
        K_yy[i, j] = SE_kernel(data_x[i], data_x[j], sigma_f=SIG_F, sigma_l=SIG_L) + SIG_N**2
K_yy_inv = np.linalg.inv(K_yy)

for idx, x in enumerate(sample_X):
    x = np.array([x])

    K_fy = np.zeros((x.shape[0], data_x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(data_x.shape[0]):
            K_fy[i, j] = SE_kernel(x[i], data_x[j], sigma_f=SIG_F, sigma_l=SIG_L)

    K_ff = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            K_ff[i, j] = SE_kernel(x[i], x[j], sigma_f=SIG_F, sigma_l=SIG_L)

    sample_mean[idx] = conditional_mean(K_yy_inv, K_fy, Y, m_y, m_f)[0]
    sample_var[idx] = conditional_variance(K_ff, K_fy, K_yy_inv)[0]

print(sample_mean)
print(sample_var)
plt.figure(0)
plt.plot(sample_X, sample_mean, color='r')
plt.plot(sample_X, sample_mean + 2 * sample_var, color='b')
plt.plot(sample_X, sample_mean - 2 * sample_var, color='b')
plt.show()


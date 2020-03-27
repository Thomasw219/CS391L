import numpy as py
import matplotlib.pyplot as plt

def SE_kernel(x_1, x_2, sigma_f=10, sigma_l=1):
    arg = -1 / (2 * sigma_l**2) + np.linalg.norm(x_1 - x_2)**2
    return sigma_f * np.exp(arg)

def conditional_mean(K_yy, K_fy, y, m_y, m_f):
    transform = np.matmul(K_fy, np.linalg.inv(K_yy))
    return np.matmul(transform, y - m_y) + m_f

def conditional_variance(K_ff, K_fy, K_yy):
    temp = np.matmul(np.inv(K_yy), np.transpose(K_fy))
    return K_ff - np.matmul(K_fy, temp)

plt.figure(0)

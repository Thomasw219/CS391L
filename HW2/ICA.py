import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def g(x):
    return 1 / (1 + np.exp(-1 * x))

def grad(W_i, X):
    Y = np.matmul(W_i, np.transpose(X))
    Z = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Z[i,j] = g(Y[i,j])
    A = np.matmul(np.ones(Z.shape) - 2 * Z, np.transpose(Y))
    return np.matmul(np.identity(W_i.shape[0]) + A, W_i)

def gradient_ascent(W_i, grad, eta, n, X):
    gradient = grad(W_i, X)
    return W_i + eta * gradient

def normalize(s_i):
    m = np.min(s_i)
    if m >= 0:
        s_i -= np.min(s_i)
    else:
        s_i += np.min(s_i)
    s_i /= np.max(s_i) / 1
    return s_i

def l2_loss(s_i, s_f):
    total = 0
    for i in range(s_i.shape[0]):
        total += (s_i[i] - s_f[i])**2
    return total

def average_correlation(S_i, S_f):
    m = S_i.shape[0]
    total = 0
    for i in range(S_i.shape[0]):
        corrs = []
        for j in range(S_f.shape[0]):
            corrs.append(np.abs(np.corrcoef(S_i[i], S_f[j])[0,1]))
        total += max(corrs)
        max_idx = np.argmax(corrs)
        S_f = np.delete(S_f, max_idx, axis=0)
    return total / S_i.shape[0]

sounds = loadmat('data/sounds.mat')['sounds']

test = loadmat('data/icaTest.mat')

num_sounds = 3
iterations = 25000
eta = 0.01
check_frequency = 250

np.random.seed(1)
s_idx = np.random.choice(sounds.shape[0], size=num_sounds)
A = np.random.rand(num_sounds, num_sounds) * 2 - np.ones((num_sounds, num_sounds))
U = sounds[s_idx,:]
X = np.matmul(A, U)
W = np.random.rand(A.shape[0], A.shape[1]) / 10
idx = np.arange(1, U.shape[1] + 1)

batch_sizes = [1, 16, 32, 64, 128]
correlations = {}
for n in batch_sizes:
    W_i = W
    correlations[n] = []
    for i in range(iterations):
        indices = np.random.choice(X.shape[1], size=n)
        batch = np.transpose(X[:,indices])
        W_i = gradient_ascent(W_i, grad, eta, n, batch)

        if i % check_frequency == 0:
            S = np.matmul(W_i, X)
            corr = average_correlation(U, S)
            correlations[n].append(corr)

            print("Iteration " + str(i))
            print(corr)
            """
            plt.figure(i)
            plt.subplot(6, 1, 1)
            plt.plot(idx, U[0])
            plt.subplot(6, 1, 2)
            plt.plot(idx, U[1])
            plt.subplot(6, 1, 3)
            plt.plot(idx, U[2])
            plt.subplot(6, 1, 4)
            plt.plot(idx, S[0])
            plt.subplot(6, 1, 5)
            plt.plot(idx, S[1])
            plt.subplot(6, 1, 6)
            plt.plot(idx, S[2])
            plt.savefig("./figures/" + str(i) + ".png")
            plt.close()
            """

colors = ['r', 'b', 'g', 'm', 'c']
p_idx = np.arange(0, iterations, check_frequency)
plt.figure(0)
for i, n in enumerate(batch_sizes):
    plt.plot(p_idx, correlations[n], label=str(n), color=colors[i])
plt.legend()
plt.savefig("./figures/batch_size_experiment_" + str(num_sounds) + "_sounds.png")

"""
plt.figure(iterations)
plt.subplot(6, 1, 1)
plt.plot(idx, U[0])
plt.subplot(6, 1, 2)
plt.plot(idx, U[1])
plt.subplot(6, 1, 3)
plt.plot(idx, U[2])
plt.subplot(6, 1, 4)
plt.plot(idx, normalize(S[0]))
plt.subplot(6, 1, 5)
plt.plot(idx, normalize(S[1]))
plt.subplot(6, 1, 6)
plt.plot(idx, normalize(S[2]))
#plt.savefig("./figures/" + str(iterations) + ".png")
plt.show()
"""

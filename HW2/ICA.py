import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def g(x):
    return 1 / (1 + np.exp(-1 * x))

W_i_inv = None
W_i_old = None
def grad(W_i, x):
    global W_i_inv
    global W_i_old
    a = np.zeros((W_i.shape[0],))
    for i, w_i in enumerate(W_i):
        a[i] = 1 - 2 * g(np.inner(w_i, x))
    if not np.array_equal(W_i, W_i_old):
        W_i_old = W_i
        W_i_inv = np.linalg.inv(np.transpose(W_i))
    return np.outer(a, x) + W_i_inv

def gradient_ascent(W_i, grad, eta, n, X):
    sum_grad = np.zeros(W_i.shape)
    for x in X:
        sum_grad += grad(W_i, x)
    ave_grad = sum_grad / n
    return W_i + eta * ave_grad

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

def total_loss(S_i, S_f):
    m = S_i.shape[0]
    total = 0
    for i in range(m):
        losses = []
        for j in range(m):
            losses.append(l2_loss(normalize(S_i[i]), normalize(S_f[j])))
        total += min(losses)
    return total

sounds = loadmat('data/sounds.mat')['sounds']

test = loadmat('data/icaTest.mat')

np.random.seed(1)
A = test['A']
U = test['U']
X = np.matmul(A, U)
W = np.random.rand(A.shape[0], A.shape[1])
idx = np.arange(1, U.shape[1] + 1)
n = 20
iterations = 1000000
eta = 0.01

least_loss = np.inf
for i in range(iterations):
    indices = np.random.choice(U.shape[1], size=n)
    batch = np.transpose(U[:,indices])
    W = gradient_ascent(W, grad, eta, n, batch)
    S = np.matmul(W, X)
    l = total_loss(U, S)

    if l < least_loss:
        least_loss = l
        print(l)

        """
        plt.figure(i)
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
        plt.savefig("./figures/" + str(i) + ".png")
        plt.close()
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
plt.savefig("./figures/" + str(iterations) + ".png")
plt.close()

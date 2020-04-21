import numpy as np
import matplotlib.pyplot as plt
import gzip

image_size = 28
num_pixels = image_size**2

nrows = 3
ncols = 5
tot = nrows * ncols

N = 10

def load_data(s, num_images):
    f = gzip.open('../HW1/' + s + '_set/' + s + '-images-idx3-ubyte.gz','r')

    f.read(16)
    im_buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(im_buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, num_pixels, 1)
    """
    image = np.asarray(data[0]).squeeze()
    plt.imshow(image)
    plt.show()
    """
    f = gzip.open('../HW1/' + s + '_set/' + s + '-labels-idx1-ubyte.gz','r')
    f.read(8)
    la_buf = f.read(1 * num_images)
    labels = np.frombuffer(la_buf, dtype=np.uint8).astype(np.int64)
    onehot = np.zeros((num_images, N, 1))
    for i, l in enumerate(labels):
        onehot[i, l, 0] = 1
    return data, onehot

train_images, train_labels = load_data('train', 60000)
test_images, test_labels = load_data('t10k', 10000)

BIG_NUM = 1000000
SMOL_NUM = 

def L(y, x):
    s = 0
    for i, x_i in enumerate(x):
        y_i = y[i]
        s += -1 * (y_i * np.log(x_i) + (1 - y_i) * np.log(1 - x_i))
    return np.asscalar(s)

def dLdx(y, x):
    d = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y_i = y[i]
        if np.asscalar(x_i) == y_i:
            d[i] = 0
        else:
            d[i] = np.asscalar(-1 * y_i / x_i + (1 - y_i)/(1 - x_i))
            if d[i] == -1 * np.inf:
                d[i] = -1 * BIG_NUM
            elif d[i] == np.inf:
                d[i] = BIG_NUM
    return d

def sig(x):
    return 1 / (1 + np.exp(-1 * x))

def dsigdx(x):
    return sig(x) * (1 - sig(x))

def sig_act(x):
    o = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        o[i] = sig(x_i)
    return o

def dsig_actdx(x):
    o = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        o[i] = np.asscalar(dsigdx(x_i))
    return o

def ReLU(x):
    return max(0, np.asscalar(x))

def dReLUdx(x):
    if np.asscalar(x) > 0:
        return 1
    else:
        return 0

def ReLU_act(x):
    o = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        o[i] = ReLU(x_i)
    return o

def dReLU_actdx(x):
    o = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        o[i] = dReLUdx(x_i)
    return o

h = 100
W_0 = np.random.random_sample(size=(h, num_pixels))
W_1 = np.random.random_sample(size=(10, h))

def forward(x0):
    y0 = np.matmul(W_0, x0)
    x1 = ReLU_act(y0)
    y1 = np.matmul(W_1, x1)
    x2 = sig_act(y1)
    return x2

def to_diagonal(x):
    mat = np.zeros((x.shape[0], x.shape[0]))
    for i, x_i in enumerate(x):
        mat[i, i] = x_i
    return mat

def compute_derivative_matrices(x0, y):
    y0 = np.matmul(W_0, x0)
    x1 = ReLU_act(y0)
    y1 = np.matmul(W_1, x1)
    x2 = sig_act(y1)
    l2 = dLdx(x2, y)
    L2 = to_diagonal(l2)
    l1 = np.matmul(np.transpose(W_1), np.matmul(L2, dsig_actdx(y1)))

    dW_1 = np.zeros(W_1.shape)
    for i in range(dW_1.shape[0]):
        for j in range(dW_1.shape[1]):
            dW_1[i, j] = np.asscalar(l2[i] * x1[i] * dsigdx(np.dot(W_1[i, :], x1)))

    dW_0 = np.zeros(W_0.shape)
    for i in range(dW_0.shape[0]):
        for j in range(dW_0.shape[1]):
            dW_0[i, j] = np.asscalar(l1[i] * x0[i] * dReLUdx(np.dot(W_0[i, :], x0)))

    return dW_0, dW_1


print(compute_derivative_matrices(train_images[0], train_labels[0]))

"""

example_y = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (1, 10))
example_x = np.reshape(np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.5, 0.1, 0.1]), (1, 10))

print(L(example_y, example_x))
print(dLdx(example_y, example_x).shape)
"""

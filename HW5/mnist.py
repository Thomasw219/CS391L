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

BIG_NUM = 10000000000000000
SMOL_NUM = 0.000000000000001

def MSE(y, x):
    s = 0
    for i, x_i in enumerate(x):
        y_i = y[i]
        s += np.power((y_i - x_i), 2)
    return np.asscalar(s)

def dMSEdx(y, x):
    d = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y_i = y[i]
        d[i] = np.asscalar(x_i - y_i)
    return d

def CE(y, x):
    s = 0
    for i, x_i in enumerate(x):
        y_i = y[i]
        s += -1 * (y_i * np.log(x_i + SMOL_NUM) + (1 - y_i) * np.log(1 - x_i + SMOL_NUM))
    return np.asscalar(s)

def dCEdx(y, x):
    d = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y_i = y[i]
        d[i] = y_i * np.asscalar(-1 / (x_i + SMOL_NUM)) + (1 - y_i) * np.asscalar(1/(1 - x_i + SMOL_NUM))
    return d

def CCE(y, x):
    s = 0
    for i, x_i in enumerate(x):
        y_i = y[i]
        s += -1 * (y_i * np.log(x_i + SMOL_NUM))
    return np.asscalar(s)

def dCCEdx(y, x):
    d = np.zeros(x.shape)
    for i, x_i in enumerate(x):
        y_i = y[i]
        d[i] = y_i * np.asscalar(-1 / (x_i + SMOL_NUM))
    return d

def SM(x, idx=None):
    s = 0
    for i, x_i in enumerate(x):
        s += np.asscalar(np.exp(x_i))
    if idx is not None:
        return np.asscalar(np.exp(x[idx])) / s
    else:
        o = np.zeros(x.shape)
        for i, x_i in enumerate(x):
            o[i] = np.asscalar(np.exp(x_i)) / s
        return o

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def dSMdz(z, i, j):
    return SM(z, idx=i) * (delta(i, j) - SM(z, idx=j))

def dSMdx(x, W, l):
    d = np.zeros(x.shape)
    m = W.shape[0]
    mat = np.zeros((m, m))
    z = np.matmul(W, x)
    for i in range(m):
        for j in range(m):
            mat[i, j] = dSMdz(z, i, j)
    for i in range(x.shape[0]):
        d[i] = np.matmul(np.transpose(l), np.matmul(mat, np.reshape(W[:, i], (m, 1))))
    return d

def sig(x):
    real = 1 / (1 + np.exp(-1 * x))
    if real == 0:
        return SMOL_NUM
    elif real == 1:
        return 1 - SMOL_NUM
    else:
        return np.asscalar(real)

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
        o[i] = dsigdx(x_i)
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

h = 200
W_0 = np.random.randn(h, num_pixels) * (1 / 20)
W_1 = np.random.randn(10, h) * (1 / 20)
ETA = 0.005

def forward(x0):
    y0 = np.matmul(W_0, x0)
#    x1 = ReLU_act(y0)
    x1 = sig_act(y0)
    y1 = np.matmul(W_1, x1)
#    x2 = sig_act(y1)
    x2 = SM(y1)
    return x2

def to_diagonal(x):
    mat = np.zeros((x.shape[0], x.shape[0]))
    for i, x_i in enumerate(x):
        mat[i, i] = x_i
    return mat

def compute_derivative_matrices(x0, y):
    y0 = np.matmul(W_0, x0)
#    x1 = ReLU_act(y0)
    x1 = sig_act(y0)
    y1 = np.matmul(W_1, x1)
#    x2 = sig_act(y1)
    x2 = SM(y1)
#    l2 = dCEdx(y, x2)
    l2 = dCCEdx(y, x2)
#    L2 = to_diagonal(l2)
#    l1 = np.matmul(np.transpose(W_1), np.matmul(L2, dsig_actdx(y1)))
    l1 = dSMdx(x1, W_1, l2)
#print(l2)
    """
    print('y1', np.mean(y1))
    print('y0', np.mean(y0))
    print('l2', l2)
    """
#    print(x2)
#    print(y)

    dW_1 = np.zeros(W_1.shape)
    for i in range(dW_1.shape[0]):
        for j in range(dW_1.shape[1]):
            dW_1[i, j] = np.asscalar(l2[i] * x1[j] * dsigdx(np.dot(W_1[i, :], x1)))

    dW_0 = np.zeros(W_0.shape)
    for i in range(dW_0.shape[0]):
        for j in range(dW_0.shape[1]):
#            dW_0[i, j] = np.asscalar(l1[i] * x0[j] * dReLUdx(np.dot(W_0[i, :], x0)))
            dW_0[i, j] = np.asscalar(l1[i] * x0[j] * dsigdx(np.dot(W_0[i, :], x0)))
            """
            if dW_0[i, j] != 0:
                print(dW_0[i, j])
            """

    return dW_0, dW_1

def update_matrices(dW_0, dW_1):
    global W_0, W_1
    W_0 -= ETA * dW_0
    W_1 -= ETA * dW_1

def test_loss_accuracy(test_images, test_labels):
    n = test_images.shape[0]
    s = 0
    c = 0
    for i in range(n):
        f = forward(test_images[i])
        l = CCE(test_labels[i], f)
        s += (1 / n) * l
        if test_labels[i, np.argmax(f)] == 1:
            c += 1
    return l, c / n

NUM_ITER = 100
BATCH_SIZE = 16
#BATCH_SIZE = 1
test_images_subset = test_images[:500]
test_labels_subset = test_labels[:500]
initial_loss, initial_accuracy = test_loss_accuracy(test_images_subset, test_labels_subset)
#print(forward(train_images[0]))
#print(train_labels[0])
print('Mean test loss: {}'.format(initial_loss))
print('Mean test accuracy: {}'.format(initial_accuracy))
losses = [initial_loss]
for i in range(NUM_ITER):
    print("===============================")
    print("Episode {}".format(i+1))
    sdW_0 = np.zeros(W_0.shape)
    sdW_1 = np.zeros(W_1.shape)
    indices = np.random.choice(60000, size=BATCH_SIZE, replace=False)
#    indices = np.random.choice(1, size=BATCH_SIZE, replace=False)
    batch_images = train_images[indices]
    batch_labels = train_labels[indices]
    for j in range(BATCH_SIZE):
        dW_0, dW_1 = compute_derivative_matrices(batch_images[j], train_labels[j])
        sdW_0 += (1 / BATCH_SIZE) * dW_0
        sdW_1 += (1 / BATCH_SIZE) * dW_1

    update_matrices(sdW_0, sdW_1)
#    print('dW_0', np.mean(sdW_0))
#    print('dW_1', np.mean(sdW_1))

    print(forward(batch_images[0]))
    print(batch_labels[0])

    epoch_loss, epoch_accuracy = test_loss_accuracy(test_images_subset, test_labels_subset)
    print('Mean test loss: {}'.format(epoch_loss))
    print('Mean test accuracy: {}'.format(epoch_accuracy))
    losses.append(epoch_loss)


"""

example_y = np.reshape(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (10, 1))
example_x = np.reshape(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), (10, 1))

print(CE(example_y, example_x))
print(dCEdx(example_y, example_x))
"""

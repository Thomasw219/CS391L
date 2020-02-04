import numpy as np
import matplotlib.pyplot as plt
import gzip
import sklearn

image_size = 28
num_pixels = image_size**2

nrows = 3
ncols = 5
tot = nrows * ncols

def load_data(s, num_images):
    f = gzip.open(s + '_set/' + s + '-images-idx3-ubyte.gz','r')

    f.read(16)
    im_buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(im_buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, num_pixels)
    """
    image = np.asarray(data[0]).squeeze()
    plt.imshow(image)
    plt.show()
    """
    f = gzip.open(s + '_set/' + s + '-labels-idx1-ubyte.gz','r')
    f.read(8)
    la_buf = f.read(1 * num_images)
    labels = np.frombuffer(la_buf, dtype=np.uint8).astype(np.int64)
    return data, labels

def sample_covariance(X):
    cov_mat = np.zeros((X.shape[1], X.shape[1]))
    mean = np.mean(X, axis=0)
    for x in X:
        cv = np.outer(x - mean, x - mean)
        cov_mat += cv
    cov_mat *= 1 / X.shape[0]
    return cov_mat

def transform_to_eigenspace(X, output_dim=-1):
    cov_mat = sample_covariance(X)
    w, v = np.linalg.eig(cov_mat)
    v_inv = np.linalg.inv(v)
    X_prime = np.matmul(v_inv, X.T).T
    if output_dim == -1:
        return X_prime
    else:
        return X_prime[:, :output_dim]

train_images, train_labels = load_data('train', 60000)
test_images, test_labels = load_data('t10k', 10000)

"""
cov_mat = sample_covariance(train_images)
np.save("cov_mat", cov_mat)
"""

cov_mat = np.load("cov_mat.npy")

"""
w, v = np.linalg.eig(cov_mat)
np.save("eigvalues", w)
np.save("eigvectors", v)
print(w)
"""

w = np.load("eigvalues.npy")
v = np.load("eigvectors.npy")
I = np.identity(num_pixels)

"""
plt.figure(0)
for i in range(tot):
    out = np.matmul(cov_mat, v[:, i])
    image = np.asarray(out.reshape(image_size, image_size, 1).astype(np.float32)).squeeze()
    plt.subplot(nrows, ncols, i + 1)
    plt.axis('off')
    plt.title(str(i))
    plt.imshow(image)
#    plt.savefig("figures/eigenvector" + str(i) + ".png")
plt.savefig("figures/eigenvectors.png")
plt.show()
"""

# Doesn't work :(
"""
A = np.zeros((num_pixels, num_pixels))
for i in range(num_pixels):
    new_basis_vec = v[:, i]
    for j in range(num_pixels):
        old_basis_vec = I[:, i]
        A[i, j] = np.dot(new_basis_vec, old_basis_vec)
"""

"""
v_inv = np.linalg.inv(v)
transformed_images = np.matmul(v_inv, train_images.T).T
np.save("eigenimages", transformed_images)
"""

transformed_images = np.load("eigenimages.npy")

"""
end = 100
ind = np.arange(0, end)
plt.bar(ind, w[0:end])
plt.savefig("figures/eigenvalues.png")
plt.show()
"""

"""
plt.figure(0)
Ns = [10, 50, 100, num_pixels]
for N in Ns:
#reduced_images = transform_to_eigenspace(train_images, output_dim=N)
    reduced_images = transformed_images[:, 0:N]
    reduced_v = v[:, :N]
    for i in range(tot):
        out = np.matmul(reduced_v, reduced_images[i])
        image = np.asarray(out.reshape(image_size, image_size, 1).astype(np.float32)).squeeze()
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        plt.title(str(train_labels[i]))
        plt.imshow(image)
    plt.savefig("eigendigits_" + str(N) + "_components.png")
    plt.show()
"""

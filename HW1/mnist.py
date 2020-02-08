import numpy as np
import matplotlib.pyplot as plt
import gzip
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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

def transform_to_eigenspace(X, eigenvectors=None, output_dim=-1):
    if eigenvectors is None:
        cov_mat = sample_covariance(X)
        w, v = np.linalg.eig(cov_mat)
        eigenvectors = v
        v_inv = np.linalg.inv(v)
    else:
        v_inv = np.linalg.inv(eigenvectors)
    X_prime = np.matmul(v_inv, X.T).T
    if output_dim == -1:
        return X_prime, eigenvectors
    else:
        return X_prime[:, :output_dim], eigenvectors

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
    plt.title(str(i + 1))
    plt.imshow(image)
#    plt.savefig("figures/eigenvector" + str(i) + ".png")
plt.savefig("figures/eigenvectors.png")
plt.show()
"""

"""
# Doesn't work :(
A = np.zeros((num_pixels, num_pixels))
for i in range(num_pixels):
    new_basis_vec = v[:, i]
    for j in range(num_pixels):
        old_basis_vec = I[:, i]
        A[i, j] = np.dot(new_basis_vec, old_basis_vec)
"""

v_inv = np.linalg.inv(v)
transformed_images = np.matmul(v_inv, train_images.T).T
np.save("eigenimages", transformed_images)

transformed_images = np.load("eigenimages.npy")

"""
end = 100
ind = np.arange(1, end + 1)
plt.bar(ind, w[0:end])
plt.savefig("figures/eigenvalues.png")
plt.show()
"""

"""
plt.figure(0)
Ns = [10, 50, 100, num_pixels]
for N in Ns:
#    reduced_images = transformed_images[:, 0:N]
    reduced_images, _ = transform_to_eigenspace(test_images, eigenvectors=v, output_dim=N)
#    reduced_images, _ = transform_to_eigenspace(train_images, eigenvectors=v, output_dim=N)
    reduced_v = v[:, :N]
    for i in range(tot):
        out = np.matmul(reduced_v, reduced_images[i])
#        out = np.matmul(reduced_v, np.matmul(v_inv[:N,:], test_images[i]))
        image = np.asarray(out.reshape(image_size, image_size, 1).astype(np.float32)).squeeze()
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        plt.title(str(test_labels[i]))
#        plt.title(str(train_labels[i]))
        plt.imshow(image)
#    plt.savefig("eigendigits_" + str(N) + "_components.png")
    plt.savefig("eigendigits_" + str(N) + "_components_test_images.png")
    plt.show()
"""

"""
neighbor_nums = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
#train_scores = [1.0, 0.9857333333333334, 0.9819166666666667, 0.975, 0.9673833333333334, 0.9536333333333333, 0.9413166666666667, 0.9266166666666666, 0.8987666666666667, 0.8662333333333333, 0.8187166666666666, 0.72735]
test_scores = [0.9691, 0.9627, 0.9688, 0.9665, 0.9625, 0.9534, 0.944, 0.9289, 0.9039, 0.8731, 0.8281, 0.7426]
eigen_test_scores = [0.9716, 0.9675, 0.9727, 0.9713, 0.9665, 0.9579, 0.95, 0.9364, 0.9128, 0.8887, 0.8474, 0.7688]
#train_scores = []
#test_scores = []
plt.figure(0)
#reduced_images, _ = transform_to_eigenspace(train_images, output_dim=100)
reduced_images = transformed_images[:, 0:100]
reduced_test_images, _ = transform_to_eigenspace(test_images, eigenvectors=v, output_dim=100)
for num_neighbors in neighbor_nums:
    print("{} neighbors started".format(num_neighbors))
    model = KNeighborsClassifier(n_neighbors=num_neighbors, n_jobs=-1)
    model.fit(reduced_images, train_labels)
    train_scores.append(model.score(reduced_images, train_labels))
    print(train_scores)
    test_scores.append(model.score(reduced_test_images, test_labels))
    print(test_scores)
    print("{} neighbors done".format(num_neighbors))
plt.plot(neighbor_nums, test_scores, c='red', label='Normal image nearest neigbors')
plt.plot(neighbor_nums, eigen_test_scores, c='blue', label='Eigen image nearest neighbors')
plt.xscale("log")
plt.xlabel("Number of Neighbors")
plt.ylabel("Test Set Accuracy")
plt.legend()
plt.savefig("figures/knn_100_eigenvectors.png")
"""

"""
model = LogisticRegression(n_jobs=-1, max_iter=10000)
print("Start training")
model.fit(train_images, train_labels)
print("End training")
print(model.score(train_images, train_labels))
print(model.score(test_images, test_labels))
print("End normal image scoring")
"""

"""
num_dimensions = [num_pixels, 500, 200, 100, 50, 20, 10]
normal_train_error = np.ones((len(num_dimensions),)) * 0.9339166666666666
normal_test_error = np.ones((len(num_dimensions),)) * 0.9255
train_scores = [0.94465, 0.9415, 0.9309333333333333, 0.9222666666666667, 0.9091, 0.8751833333333333, 0.8010166666666667]
test_scores = [0.9211, 0.9228, 0.9266, 0.9221, 0.9125, 0.8801, 0.8086]
#train_scores = []
#test_scores = []
plt.figure(0)

model = LogisticRegression(n_jobs=-1, max_iter=10000)
for d in num_dimensions:
    print("{} dimensions started".format(d))
    reduced_images = transformed_images[:, 0:d]
    reduced_test_images, _ = transform_to_eigenspace(test_images, eigenvectors=v, output_dim=d)
    model.fit(reduced_images, train_labels)
    train_scores.append(model.score(reduced_images, train_labels))
    print(train_scores)
    test_scores.append(model.score(reduced_test_images, test_labels))
    print(test_scores)
    print("{} dimensions done".format(d))

#plt.plot(num_dimensions, train_scores, c='blue', label='Eigenspace Training')
plt.plot(num_dimensions, test_scores, c='red', label='Eigenspace Logistic Regression')
#plt.plot(num_dimensions, normal_train_error, c='black', label='Baseline Training')
plt.plot(num_dimensions, normal_test_error, c='grey', label='Baseline Logistic Regression')
plt.xlabel("Number of Dimensions")
plt.ylabel("Test Set Accuracy")
plt.legend()
plt.savefig("figures/logistic_regression_eigenvectors.png")
plt.show()
"""

"""
training_set_size = [1000, 2000, 5000, 10000, 20000, 50000]
test_scores = []
eigen_test_scores = []
test_scores = [0.8786, 0.9077, 0.935, 0.948, 0.9551, 0.9617]
eigen_test_scores = [0.8875, 0.913, 0.9417, 0.9523, 0.9601, 0.9674]
plt.figure(0)
for n in training_set_size:
    idx = np.random.randint(60000, size=n)
    train_subset = train_images[idx, :]
    label_subset = train_labels[idx]
    reduced_images, v_prime = transform_to_eigenspace(train_subset, output_dim=100)
    reduced_test_images, _ = transform_to_eigenspace(test_images, eigenvectors=v_prime, output_dim=100)
    print("{} training points started".format(n))
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(train_subset, label_subset)
    test_scores.append(model.score(test_images, test_labels))
    print(test_scores)
    model.fit(reduced_images, label_subset)
    eigen_test_scores.append(model.score(reduced_test_images, test_labels))
    print(eigen_test_scores)
    print("{} training points done".format(n))
plt.plot(training_set_size, test_scores, c='red', label='Normal image nearest neigbors')
plt.plot(training_set_size, eigen_test_scores, c='blue', label='Eigen image nearest neighbors')
plt.xscale("log")
plt.xlabel("Images in Training Set")
plt.ylabel("Test Set Accuracy")
plt.legend()
plt.savefig("figures/knn_vary_training_size.png")
"""

num_dimensions = [500, 200, 100, 50, 20, 10]
normal_score = np.ones((len(num_dimensions),)) * 0.9688
test_scores = []
plt.figure(0)

model = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
for d in num_dimensions:
    print("{} dimensions started".format(d))
    reduced_images = transformed_images[:, 0:d]
    reduced_test_images, _ = transform_to_eigenspace(test_images, eigenvectors=v, output_dim=d)
    model.fit(reduced_images, train_labels)
#    train_scores.append(model.score(reduced_images, train_labels))
#    print(train_scores)
    test_scores.append(model.score(reduced_test_images, test_labels))
    print(test_scores)
    print("{} dimensions done".format(d))

plt.plot(num_dimensions, test_scores, c='red', label='Eigenvector nearest neigbors')
plt.plot(num_dimensions, normal_score, c='grey', label='Baseline nearest neighbors')
plt.xlabel("Number of Dimensions")
plt.ylabel("Test Set Accuracy")
plt.legend()
plt.savefig("figures/knn_vary_dimensions.png")
plt.show()


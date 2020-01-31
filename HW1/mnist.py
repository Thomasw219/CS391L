import numpy as np
import matplotlib.pyplot as plt
import gzip

image_size = 28
num_pixels = image_size**2

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

def sample_covariance(images):
    cov_mat = np.zeros((num_pixels, num_pixels))
    mean = np.mean(images, axis=0)
    for image in images:
        cv = np.outer(image - mean, image - mean)
        cov_mat += cv
        """
        image = np.asarray(cv.reshape(num_pixels, num_pixels, 1).astype(np.float32)).squeeze()
        plt.imshow(image)
        plt.show()
        """
    cov_mat *= 1 / images.shape[0]
    return cov_mat

train_images, train_labels = load_data('train', 60000)
test_images, test_labels = load_data('t10k', 10000)

"""
cov_mat = sample_covariance(train_images)
np.save("cov_mat", cov_mat)
"""
cov_mat = np.load("cov_mat.npy")
print(cov_mat.shape)
w, v = np.linalg.eig(cov_mat)
print(w)
print((np.matmul(cov_mat, v[0]).shape))
for i in range(10):
    image = np.asarray((np.matmul(cov_mat, v[i])).reshape(image_size, image_size, 1).astype(np.float32)).squeeze()
#image = np.asarray(cov_mat).squeeze()
    plt.imshow(image)
    plt.show()

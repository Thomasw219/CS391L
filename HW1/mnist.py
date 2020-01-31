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
    for image in images:
        cov_mat += image * image.T

train_images, train_labels = load_data('train', 60000)
test_images, test_labels = load_data('t10k', 10000)

print(sample_covariance(train_images))

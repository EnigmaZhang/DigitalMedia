"""
@Author: Enigma Zhang

@Description:
This program is to use K-L Transform or PCA on a corel dataset image
and implement LBG-VQ algorithm to a image.

"""

import shutil

from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans, vq

P = 1


def read_data(path):
    image = Image.open(path).convert("L")
    data = np.asarray(image)
    return data


def similar(x, y):
    for i in zip(x, y):
        if abs(i[1] - i[0]) > 0.01:
            return False
    return True


def read_vq_data(path):
    image = Image.open(path)
    data = np.asarray(image).astype("float")
    origin_shape = data.shape
    data = data.reshape((data.shape[0] * data.shape[1]), data.shape[2])
    return data, origin_shape


if __name__ == "__main__":
    path = r"./corel/Corel100类库/0_1.jpg"
    data = read_data(path)
    # No rgb channel.
    print("Image shape:", data.shape)
    # Col as a var, row as attr.
    M = np.mean(data, axis=0)
    data = data - M

    C = np.cov(data.T)

    eigenvalue, eigenvetcor = np.linalg.eig(C)
    # Get the index of reversed sorted eigenvalue
    sorted_index = np.argsort(-eigenvalue)[:P]
    # Get the first p vectors to get the projection matrix.

    Projection = eigenvetcor[:, sorted_index]

    N = np.dot(data, Projection)
    print(N.shape)
    print(N)

    pca = PCA(n_components=1)
    new_N = pca.fit_transform(read_data(path))
    # Test if my calculation is same as scikit-learn.
    print(similar(N, -new_N))

    shutil.copy(path, r"./out/origin.jpg")
    data, origin_shape = read_vq_data(path)
    vq_k = [2, 10, 256]
    for k in vq_k:
        # Generate code_book
        code_book, _ = kmeans(data, k)
        code, _ = vq(data, code_book)
        result = code_book[code, :]
        result = result.reshape(origin_shape).astype("uint8")
        image = Image.fromarray(result)
        image.save(r"./out/result_{}.jpg".format(k))

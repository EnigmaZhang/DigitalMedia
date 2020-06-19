import os
import time
import random

import numpy as np
from bitarray import bitarray
from PIL import Image

"""
@Author: Enigma Zhang

@Description:
This program is to design a LSH algorithm, distance function is hamming.

"""


class LSH:
    """
    :parameter bit_length:
    The length of hash binary.
    :parameter input_dim:
    The size of image vector.
    """

    def __init__(self, bit_length, input_dim):
        self.bit_length = bit_length
        self.input_dim = input_dim
        # Use uniform plane to generate hash.
        self.uniform_plane = np.random.randn(self.bit_length, self.input_dim)
        self.hashtable = {}

    def _hash(self, plane, input_point):
        """
        Generate hash.
        :return: return binary hash.
        """
        input_point = np.array(input_point)
        projections = np.dot(plane, input_point)
        return "".join(["1" if i > 0 else "0" for i in projections])

    def index(self, input_point):
        input_point_hash = self._hash(self.uniform_plane, input_point)
        if input_point_hash in self.hashtable:
            self.hashtable[input_point_hash].append(tuple(input_point))
        else:
            self.hashtable[input_point_hash] = [tuple(input_point)]

    # Get the first_nums of result.
    def query(self, query_point, first_nums):
        query_point_hash = self._hash(self.uniform_plane, query_point)
        result = [i for i in self.hashtable.keys()]
        result.sort(key=lambda x: LSH.hamming_dist(x, query_point_hash))
        result = result[:10]
        ret = []
        for i in result:
            ret.extend(self.hashtable[i])
        return ret

    @staticmethod
    def hamming_dist(x, y):
        return (bitarray(x) ^ bitarray(y)).count()


bit_length = 20     # dim / 0.7 to bin
dim = 126 * 187 * 3
first = 10
lsh = LSH(bit_length, dim)
file_count = 0
for _, _, files in os.walk(r"./corel/Corel100类库/"):
    for f in files:
        if f.endswith(".jpg"):
            data = np.asarray(Image.open(r"./corel/Corel100类库/" + f))
            if data.shape[0] * data.shape[1] * data.shape[2] == dim:
                lsh.index(data.reshape(dim))
                file_count += 1
            if file_count == 1000:
                break

positive = []
negative = []
for _, _, files in os.walk(r"./corel/Corel100类库/"):
    for i in range(100):
        index = random.randint(0, 2000)
        if files[index].endswith(".jpg"):
            data = np.asarray(Image.open(r"./corel/Corel100类库/" + files[index]))
            if data.shape[0] * data.shape[1] * data.shape[2] == dim:
                if index < 1000:
                    positive.append(data.reshape(dim))
                else:
                    negative.append(data.reshape(dim))

TP = 0
FN = 0
TN = 0
FP = 0
t1 = time.time()
for i in positive:
    result = lsh.query(i, 10)
    if tuple(i) in result:
        TP += 1
    else:
        FN += 1
for i in negative:
    result = lsh.query(i, 10)
    if tuple(i) in result:
        FP += 1
    else:
        TN += 1
t2 = time.time()
print("Time to find {}".format((t2 - t1) / 100))
print("Accuracy: {}".format((TP + TN) / 100))
print("Recall:{}".format(TP / (TP + FN)))

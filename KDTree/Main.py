"""
@Author: Enigma Zhang
@Reference: https://leileiluoluo.com/posts/kdtree-algorithm-and-implementation.html
            https://www.jianshu.com/p/7d15944290fb
@Description:
This program is build a KD-Tree and evaluate its performance.

"""

import numpy as np
from math import sqrt
from sklearn.neighbors import KDTree as sklearn_KDTree
import random
import time

class Node:
    def __init__(self, point, dim):
        self.point = point
        self.dim = dim
        self.left = None
        self.right = None

    def set_left(self, left_point):
        self.left = left_point

    def set_right(self, right_point):
        self.right = right_point

    def is_leaf(self):
        return self.left is None and self.right is None

    def __str__(self):
        left_value = self.left.point if self.left else self.left
        right_value = self.right.point if self.right else self.right
        return "value: {}, left: {}, right: {}".format(self.point, left_value, right_value)


class KDTree:
    def __init__(self, points):
        # Select the bigger var dim as first sliced.
        var_x = np.var(np.asarray([x[0] for x in points]).astype("float32"))
        var_y = np.var(np.asarray([x[1] for x in points]).astype("float32"))
        self.point_num = 1
        self.points = points
        self.dim_chooser = [1, 0]
        self.first_dim = 0 if var_x > var_y else 1
        sort_points = sorted(self.points, key=lambda x: x[self.first_dim])
        middle = len(sort_points) // 2
        self.root = Node(sort_points[middle], self.first_dim)
        self._build_node(node=self.root, left_points=sort_points[:middle], right_points=sort_points[middle + 1:],
                         dim=self.dim_chooser[self.first_dim])

    def _build_node(self, node, left_points, right_points, dim):
        if len(left_points) > 0:
            sort_points = sorted(left_points, key=lambda x: x[dim])
            middle = len(sort_points) // 2
            node.left = Node(sort_points[middle], dim)
            self.point_num += 1
            self._build_node(node=node.left, left_points=sort_points[:middle], right_points=sort_points[middle + 1:],
                             dim=self.dim_chooser[dim])

        if len(right_points) > 0:
            sort_points = sorted(right_points, key=lambda x: x[dim])
            middle = len(sort_points) // 2
            node.right = Node(sort_points[middle], dim)
            self.point_num += 1
            self._build_node(node=node.right, left_points=sort_points[:middle], right_points=sort_points[middle + 1:],
                             dim=self.dim_chooser[dim])

    def print_tree(self):
        print("Nodes num: {}".format(self.point_num))
        print("First dim: {}".format(self.first_dim))
        nodes = [self.root]
        for i in range(self.point_num):
            if nodes[i].left:
                nodes.append(nodes[i].left)
            if nodes[i].right:
                nodes.append(nodes[i].right)
        for i in nodes:
            print(i.point)

    def search(self, point):
        search_point = [self.root]
        now_node = self.root
        while not now_node.is_leaf():
            if point[now_node.dim] <= now_node.point[now_node.dim]:
                if now_node.left:
                    search_point.append(now_node.left)
                    now_node = now_node.left
                else:
                    break
            else:
                if now_node.right:
                    search_point.append(now_node.right)
                    now_node = now_node.right
                else:
                    break
        nearest_distance = self.euclid_distance(point, now_node.point)
        while search_point:
            # Backward is nearer?
            back_node = search_point[-1]
            if self.euclid_distance(back_node.point, point) < nearest_distance:
                nearest_distance = self.euclid_distance(back_node.point, point)
                now_node = back_node

            # The line split vertical.
            if back_node.dim == 0:
                if abs(back_node.point[0] - point[0]) < nearest_distance:
                    # The other side might have nearer point, so must search sub-nodes.
                    if back_node.left and back_node.left not in search_point:
                        search_point.append(back_node.left)
                    if back_node.right and back_node.right not in search_point:
                        search_point.append(back_node.right)
            # The line split is horizontal
            else:
                if abs(back_node.point[1] - point[1]) < nearest_distance:
                    # The other side might have nearer point, so must search sub-nodes.
                    if back_node.left and back_node.left not in search_point:
                        search_point.append(back_node.left)
                    if back_node.right and back_node.right not in search_point:
                        search_point.append(back_node.right)
            search_point.remove(back_node)

        return now_node


    @staticmethod
    def euclid_distance(a, b):
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


BJ_data = []
BJ_data_sk = []
with open(r"./data/BJ/real.txt") as f:
    for line in f.readlines():
        BJ_data.append(tuple(float(i) for i in line.strip().split(" ")[-2:]))
        BJ_data_sk.append([float(i) for i in line.strip().split(" ")[-2:]])

tree = KDTree(BJ_data)
test_tree = sklearn_KDTree(np.asarray(BJ_data_sk))
print(tree.search((116.42, 39.97)))
result_index = test_tree.query(np.asarray([[116.42, 39.97]]), return_distance=False)[0][0]
print(BJ_data_sk[result_index])

CA_data = []
CA_data_sk = []
with open(r"./data/CA/real.txt") as f:
    for line in f.readlines():
        CA_data.append(tuple(float(i) for i in line.strip().split(" ")[-2:]))
        CA_data_sk.append([float(i) for i in line.strip().split(" ")[-2:]])
tree = KDTree(CA_data)
test_tree = sklearn_KDTree(np.asarray(CA_data_sk))
print(tree.search((-122.42, 37.89)))
result_index = test_tree.query(np.asarray([[-122.42, 37.89]]), return_distance=False)[0][0]
print(CA_data_sk[result_index])

test_points = []
for count in range(10000):
    test_points.append([-122.9 + random.random(), 37 + random.random()])

t1 = time.time()
for test_point in test_points:
    tree.search(test_point)
t2 = time.time()
print("My tree time: {}".format(t2 - t1))

t1 = time.time()
for test_point in test_points:
    test_tree.query(np.asarray([test_point]))
t2 = time.time()
print("Sklearn tree time: {}".format(t2 - t1))
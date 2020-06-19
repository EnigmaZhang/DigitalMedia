import cv2
import numpy as np
import os
from features import MOPSFeatureDescriptor, RatioFeatureMatcher, HarrisKeypointDetector

"""
@Author: Enigma Zhang

@Description:
This program is to find the most similar image with image descriptor.
This program uses two method, and the first is designed myself as a CV homework.

"""

# Harris + MOPS + RatioSSD, most match as most similar.
mops = MOPSFeatureDescriptor()
mops_feature = {}
harris = HarrisKeypointDetector()
matcher = RatioFeatureMatcher()
for dirpath, _, files in os.walk(r"./data/train/0/"):
    for f in files:
        if f.endswith(".jpg"):
            image = np.asarray(cv2.imread(os.path.join(dirpath, f), cv2.IMREAD_COLOR))
            mops_feature[f] = mops.describeFeatures(image, harris.detectKeypoints(image))

test_image = np.asarray(cv2.imread(r"./data/test/0/0_3.jpg", cv2.IMREAD_COLOR))
test_feature = mops.describeFeatures(test_image, harris.detectKeypoints(test_image))
max_f = max(mops_feature.items(), key=lambda x: len(matcher.matchFeatures(x[1], test_feature)))
match_image = cv2.imread(r"./data/train/0/" + max_f[0])
cv2.imwrite(r"./data/match/MOPS.jpg", match_image)

# Sift + BFMatch, least distance as most similar.
sift = cv2.xfeatures2d.SIFT_create()
sift_feature = {}
bf = cv2.BFMatcher()
for dirpath, _, files in os.walk(r"./data/train/0/"):
    for f in files:
        if f.endswith(".jpg"):
            image = cv2.imread(os.path.join(dirpath, f), cv2.IMREAD_COLOR)
            sift_feature[f] = sift.detectAndCompute(image, None)[1]
test_image = cv2.imread(r"./data/test/0/0_3.jpg", cv2.IMREAD_COLOR)
test_feature = sift.detectAndCompute(test_image, None)[1]
max_f = min(sift_feature.items(),
            key=lambda x: sum([i.distance for i in bf.match(x[1], test_feature)]))
match_image = cv2.imread(r"./data/train/0/" + max_f[0])
cv2.imwrite(r"./data/match/SIFT.jpg", match_image)

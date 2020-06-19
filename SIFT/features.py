import math
import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def saveHarrisImage(self, harrisImage, srcImage):
        '''
        Saves a visualization of the harrisImage, by overlaying the harris
        response image as red over the srcImage.

        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]  # 数组前2个维度

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        ix = np.zeros(srcImage.shape[:2])  # 定义x方向导数
        iy = np.zeros(srcImage.shape[:2])  # 定义Y方向导数
        ndimage.filters.sobel(srcImage, 1, ix)  # 求解x方向导数
        ndimage.filters.sobel(srcImage, 0, iy)  # 求解y方向导数

        ixx = ix ** 2
        iyy = iy ** 2
        ixy = ix * iy

        sigma = 0.5
        gixx = ndimage.filters.gaussian_filter(ixx, sigma)  # 高斯滤波 sigma=0.5
        gixy = ndimage.filters.gaussian_filter(ixy, sigma)
        giyy = ndimage.filters.gaussian_filter(iyy, sigma)

        harrisImage = gixx * giyy - gixy ** 2 - 0.1 * ((gixx + giyy) ** 2)  # R = det(H)-0.1(trace(H))^2

        for i in range(srcImage.shape[0]):  # 用arctan求梯度方向
            for j in range(srcImage.shape[1]):
                orientationImage[i, j] = np.degrees(np.arctan2(iy[i, j], ix[i, j]))

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        # self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN

        localmaxim = ndimage.filters.maximum_filter(harrisImage, (7, 7))  # 局部最大值是7*7邻域的最大值

        for i in range(harrisImage.shape[0]):
            for j in range(harrisImage.shape[1]):
                if localmaxim[i][j] == harrisImage[i][j]:
                    destImage[i][j] = True  # 如Harris值等于局部最大值，则标记为True

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for i in range(height):
            for j in range(width):
                if not harrisMaxImage[i, j]:
                    continue

                f = cv2.KeyPoint()  # 标记为True的像素点是特征点

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                f.size = 10
                f.pt = (j, i)  # 特征点的坐标(顺序反写)
                f.response = harrisImage[i][j]  # 特征点的Harris强度
                f.angle = orientationImage[i][j]  # 特征点的梯度方向

                # raise Exception("TODO in features.py not implemented")
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
        detector = cv2.ORB_create()
        return detector.detect(image, None)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output: 
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))  # 二维数组，每行是一个特征点的5*5邻域中的像素强度值

        for i, f in enumerate(keypoints):  # 枚举特征点
            y, x = f.pt  # 特征点的坐标
            y, x = int(y), int(x)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN

            extd = np.pad(grayImage, ((2, 2), (2, 2)), 'constant')  # 用0填充图像边缘
            nbhd = extd[x:x + 5, y:y + 5]  # 获取特征点的5*5邻域，因为填充过所以不是x-2:x+3
            desc[i, :] = np.reshape(nbhd, (1, -1))  # 将5*5矩阵转为一维数组,行数为1，‘-1’表示列数由元素数量决定

            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN

            y, x = f.pt
            y, x = int(y), int(x)
            # gi = np.pad(grayImage,((20,20),(20,20)),'constant')
            # gi40 = gi[x:x+41,y:y+41]
            # m1，m2，m3，m4 依次为 平移，旋转，缩放，平移
            M1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])  # 平移变换矩阵1(将旋转中心平移到坐标原点)

            ang = -np.radians(f.angle)
            M2 = np.array([[math.cos(ang), -math.sin(ang), 0],
                           [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])  # 旋转变换矩阵(将梯度方向转向正右方)
            M3 = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])  # 缩放变换矩阵(从40*40缩小至8*8,等于乘以0.2)
            M4 = np.array([[1, 0, windowSize / 2], [0, 1, windowSize / 2], [0, 0, 1]])  # 平移变换矩阵2(将窗口左上角平移到坐标原点)
            M = np.dot(np.dot(np.dot(M4, M3), M2), M1)  # 因为是左乘，所以最先运算的是最后一步变换(矩阵乘法具有可交换性)
            transMx = M[:2, :3]  # 取矩阵的前两行作为2*3仿射变换矩阵(第三行一直是0,0,1所以只取前两行)

            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix

            # destImage = cv2.warpAffine(gi40, transMx,
            #                            (windowSize, windowSize), flags=cv2.INTER_LINEAR)
            destImage = cv2.warpAffine(grayImage, transMx,
                                       (windowSize, windowSize), flags=cv2.INTER_LINEAR)  # 参数是grayImage,用整个图像代替了40*40邻域

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN

            # meanval =
            destImage -= np.mean(destImage)  # 零均值

            stdval = np.std(destImage)
            if stdval < math.pow(10, -5):
                desc[i, :] = np.zeros((1,))  # ①方差非常接近零(小于10^-5)时，全零填充行向量
            else:
                destImage /= stdval  # 单位方差是在零均值的基础上，每个元素除以标准差
                desc[i, :] = np.reshape(destImage, (1, -1))  # ②否则用降为一维的8*8图像块填充行向量
                # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        """
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6] * x + h[7] * y + h[8]

        return np.array([(h[0] * x + h[1] * y + h[2]) / d,
                         (h[3] * x + h[4] * y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        """
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN

        dis = spatial.distance.cdist(desc1, desc2, 'euclidean')  # 得到K1*K2大小的欧式距离数组
        for i in range(dis.shape[0]):  # 第i行是图一的第i个特征点与图二的所有特征点依次求欧式距离得到的一维数组
            dm = cv2.DMatch(i, np.argmin(dis[i]), dis[i, np.argmin(dis[i])])
            dm.queryIdx = i
            j = np.argmin(dis[i])  # 最小距离的下标
            dm.trainIdx = j
            dm.distance = dis[i, j]  # 最小距离
            matches.append(dm)  # 添加到matches列表中

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        """
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN

        distances = spatial.distance.cdist(desc1, desc2, 'euclidean')
        for i, distance in enumerate(distances):
            smallest_index, second_index = distance.argsort()[:2]
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = int(smallest_index)
            match.distance = float(distance[smallest_index]) / distance[second_index]
            matches.append(match)

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

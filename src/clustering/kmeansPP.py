import numpy as np
from scipy.spatial.distance import cdist
from ..utils import convertToNumpy
"""
from: https://en.wikipedia.org/wiki/K-means%2B%2B

k-means++ is an algorithm for choosing the initial values (or "seeds") for the k-means clustering algorithm.

The exact algorithm is as follows:

- Choose one center uniformly at random among the data points.
- For each data point x, compute D(x), the distance between x
    and the nearest center that has already been chosen.
- Choose one new data point at random as a new center, using 
    a weighted probability distribution where a point x is
    chosen with probability proportional to D(x)2.
- Repeat Steps 2 and 3 until k centers have been chosen.
- Now that the initial centers have been chosen, proceed using 
    standard k-means clustering.

OR

- Randomly select the first centroid from the data points.
- For each data point compute its distance from the nearest, 
previously chosen centroid.
- Select the next centroid from the data points such that the 
probability of choosing a point as centroid is directly 
proportional to its distance from the nearest, 
previously chosen centroid. 
(i.e. the point having maximum distance from the nearest 
centroid is most likely to be selected next as a centroid)
- Repeat steps 2 and 3 untill k centroids have been sampled
"""

class KMeansPP:
    def __init__(self, n_clusters=8):
        self.numberOfClusters = n_clusters
        self.centers = None

    def get_centroids(self):
        return self.centers

    def fit(self, trainData):
        trainData = convertToNumpy(trainData)
        self._initializeCenters(trainData)
        for _ in range(self.numberOfClusters - 1):
            minCentroidDistance = self._predictClusters(trainData)
            newCenter = self._computeNewCenter(trainData, minCentroidDistance)
            self._updateCenters(newCenter)
        return self

    def _initializeCenters(self, data):
        randomRow = np.random.choice(data.shape[0], 1)
        self.centers = data[randomRow]

    def _computeNewCenter(self, trainData, distanceMatrix):
        newCenterRow = np.argmax(distanceMatrix)
        return trainData[newCenterRow]

    def _updateCenters(self, newCenter):
        self.centers = np.concatenate((self.centers, [newCenter]), axis=0)

    def _predictClusters(self, data):
        l2distances = cdist(data, self.centers)
        return np.min(l2distances, axis=1)


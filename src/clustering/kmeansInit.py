import numpy as np
from scipy.spatial.distance import cdist
from utils import convertToNumpy
# from .kmeans import KMeans

class BaseInitializer:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.centers = None

    @property
    def centroids(self):
        return self.centers

    def fit(self, trainData):
        """compute centroids according to the training data

        Args:
            trainData ([np.array, pd.DataFrame or list of lists]): [data of shape (n_samples, 
                n_features) from where to compute the centroids]

        Returns:
            [KMeansPP]: [fitted KMeansPP object]
        """
        trainData = convertToNumpy(trainData)
        return self._fit(trainData)

    def _fit(self, trainData:np.ndarray):
        raise NotImplementedError('abstract method, implement _fit in the class inheriting')


class KMeansPP(BaseInitializer):
    """KmeansPP initializations generator

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

        Args:
            n_clusters (int, optional): [number of centers to be computed]. Defaults to 8.
    """

    def _fit(self, trainData:np.ndarray):
        """internal fit method. Iterates over the number of clusters to compute the distances 
        and choose the optimal cluster according to the rest.

        Args:
            trainData ([np.ndarray]): [data of shape (n_samples, 
                n_features) from where to compute the centroids]

        Returns:
            [KMeansPP]: [fitted KMeansPP object]
        """
        self._initializeCenters(trainData)
        for _ in range(self.n_clusters - 1):
            minCentroidDistance = self._predictClusters(trainData)
            newCenter = self._computeNewCenter(trainData, minCentroidDistance)
            self._updateCenters(newCenter)
        return self

    def _initializeCenters(self, data):
        """Extract the first centroid as a random instance of the data

        Args:
            data ([np.ndarray]): [array of shape (n_samples, n_features) from where a random
                row (axis=0) will be chosen as the initial cluster]
        """
        randomRow = np.random.choice(data.shape[0], 1)
        self.centers = data[randomRow]

    def _computeNewCenter(self, trainData, distanceMatrix):
        """given the trainData and the distanceMatrix, computes and extracts the instance further

        Args:
            trainData ([np.ndarray]): [(n_samples, n_features)) instance matrix]
            distanceMatrix ([np.ndarray]): [(n_centers, n_samples) distance matrix from each center to n_samples]

        Returns:
            [center]: [new center]
        """
        newCenterRow = np.argmax(distanceMatrix)
        return trainData[newCenterRow]

    def _updateCenters(self, newCenter):
        """concatenates the center to the array of generated centers

        Args:
            newCenter ([np.ndarray]): [center of shape (n_features,)]
        """
        self.centers = np.concatenate((self.centers, [newCenter]), axis=0)

    def _predictClusters(self, data):
        """assign clusters to data, computes the closest center to each data instance.

        Args:
            data ([np.ndarray]): [(n_samples, n_features) to be infered]

        Returns:
            [labels]: [(n_samples,) label array]
        """
        l2distances = cdist(data, self.centers)
        return np.min(l2distances, axis=1)


class RPKM(BaseInitializer):
    def __init__(self, n_clusters=8, max_iter=100):
        super(RPKM, self).__init__(n_clusters=n_clusters)
        self.max_iter = max_iter

    def _fit(self, data):
        from .kmeans import KMeans
        data_indexes = np.arange(data.shape[0])
        np.random.shuffle(data_indexes)
        partitions = np.array_split(data_indexes, self.n_clusters)
        representative = np.array(list(map(lambda indexes: data[indexes].mean(axis=0), partitions)))
        self.centers = representative.copy()
        old_centers = None
        iteration_idx = 0
        while self.continue_criterion(self.centers, old_centers): # stop condition: difference between centers
            old_centers = self.centers
            partitions = self._segment_partitions(partitions, data.shape[1]) # create partitions
            if len(partitions) >= data.shape[0]:
                break
            representative = np.array(list(map(lambda indexes: data[indexes].mean(axis=0), partitions)))
            self.centers = KMeans(n_clusters=self.n_clusters).fit(representative).centers
            iteration_idx += 1
            if iteration_idx >= self.max_iter:
                break
        return self

    def continue_criterion(self, new_centers, old_centers):
        if old_centers is None:
            return True
        return False

    def _segment_partitions(self, partition_indexes, d):
        positions = range(len(partition_indexes))
        for partition_idx in reversed(positions):
            subparts = np.array_split(partition_indexes[partition_idx], 2**d)
            partition_indexes.pop(partition_idx)
            partition_indexes.extend(subparts)
        return partition_indexes

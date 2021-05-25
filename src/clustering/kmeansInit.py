import numpy as np
import logging
from scipy.spatial.distance import cdist
from utils import convertToNumpy
from sklearn.cluster import KMeans


class BaseInitializer:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.distance_computations_ = 0
        self.centers = None

    @property
    def distance_computations(self):
        return self.distance_computations_

    @property
    def centroids(self):
        return self.centers

    def fit(self, trainData):
        """compute centroids according to the training data

        Args:
            trainData ([np.array, pd.DataFrame or list of lists]): [data of shape (n_samples, 
                n_features) from where to compute the centroids]

        Returns:
            [Initializer]: [fitted Initializer object]
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
        self.distance_computations_ += data.shape[0] * self.centers.shape[0]
        l2distances = cdist(data, self.centers)
        return np.min(l2distances, axis=1)


class RPKM(BaseInitializer):
    def __init__(self, n_clusters=8, max_iter=6, distance_threshold=1e-4, n_jobs=-1):
        super(RPKM, self).__init__(n_clusters=n_clusters)
        self.reset()
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def reset(self):
        self.centers = None
        self.distance_computations_ = 0

    @property
    def distance_computations(self):
        return self.distance_computations_

    def _fit(self, data):
        self.n_dim = data.shape[1]
        partitions = self.binary_partition(data, max_depth=self.max_iter+1)
        partition_meta = self.extract_meta_from_partition(partitions)

        for i, (R, cardinality) in partition_meta.items():
            if len(cardinality) < self.n_clusters:
                continue
            elif self.centers is None:
                centers_idx = np.random.choice(range(len(cardinality)), self.n_clusters, replace=False)
                self.centers = R[centers_idx]
                if self.centers.shape[0] == self.n_clusters:
                    continue

            old_centers = self.centers.copy()
            km = KMeans(
                n_clusters=self.n_clusters, 
                init=self.centers,
                n_init=1,
                algorithm='full', # lloyd
                n_jobs=self.n_jobs
            ).fit(
                R,
                sample_weight=cardinality
            )
            self.centers = km.cluster_centers_
            self.distance_computations_ += km.n_iter_ * self.n_clusters * R.shape[0]

            if self.stopCriterion(old_centers):
                break
        else:
            if self.centers is None:
                logging.warning('centers could not be initialized, falling back to random')
                randomRowIdxs = np.random.choice(data.shape[0], self.n_clusters, replace=False)
                self.centers = data[randomRowIdxs]
        return self

    def stopCriterion(self, old_centers):
        return cdist(self.centers, old_centers).diagonal().max() < self.distance_threshold

    @staticmethod
    def extract_meta_from_partition(partitions:dict):
        partition_stats = dict()
        for k, v in partitions.items():
            partition_stats[k] = (
                np.array([i.mean(axis=0) for i in v]),
                [float(i.shape[0]) for i in v]
        )
        return partition_stats

    def binary_partition(self, data, depth=0, max_depth=2, max_=None, min_=None, thresholds=None, partitions=None):
        n_dim = data.shape[1]

        if thresholds is None:
            thresholds = np.zeros((n_dim,))

        if max_ is None:
            max_ = np.ones((n_dim,))
        elif isinstance(max_, int):
            max_ = max_ * np.ones((n_dim,))

        if min_ is None:
            min_ = -1*np.ones((n_dim,))
        elif isinstance(min_, int):
            min_ = min_ * np.ones((n_dim,))

        if depth > max_depth - 1:
            return

        if partitions is None:
            partitions = dict()

        # build binary classification array for subspace
        comps = np.array([data[:, dim] > thresholds[dim] for dim in range(n_dim)]).astype(int)

        # convert bits to uint8
        hypercube_idx = np.packbits(comps, axis=0, bitorder='little').flatten()

        local_partitions = [data[hypercube_idx == i] for i in set(hypercube_idx)]
        if depth not in partitions:
            partitions[depth] = list()
        partitions[depth].extend(local_partitions)

        for i, partition in zip(set(hypercube_idx), local_partitions):
            sub_max = max_.copy()
            sub_min = min_.copy()
            sub_th = thresholds.copy()
            bin_rep = np.unpackbits(np.array([i], dtype=np.uint8), bitorder='little', count=n_dim)
            for i, b in enumerate(bin_rep):
                if b == 1:
                    sub_min[i] = sub_th[i]
                else:
                    sub_max[i] = sub_th[i]
                sub_th[i] = (sub_max[i] + sub_min[i])/2
            ret = self.binary_partition(partition, 
                                depth=depth+1, 
                                max_depth=max_depth,
                                max_=sub_max,
                                min_=sub_min,
                                thresholds=sub_th,
                                partitions=partitions)
            if isinstance(ret, dict):
                partitions = ret
        return partitions

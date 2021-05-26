import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans

# struct
class Partition:
    def __init__(self, data, max, min, thresholds):
        self.data = data
        self.max = max.astype(np.float)
        self.min = min.astype(np.float)
        self.thresholds = thresholds.astype(np.float)
        self.R = np.mean(data, axis=0)
        self.cardinality = data.shape[0]

    def __str__(self):
        return f"{self.__class__.__name__}: [min={self.min}, max={self.max}, thresholds={self.thresholds}, R={self.R}, cardinality={self.cardinality}]"

    def __repr__(self):
        return self.__str__()


class RPKM(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=8, max_iter=6, distance_threshold=1e-4, n_jobs=-1):
        self.n_clusters = n_clusters
        self.reset()
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def reset(self):
        self.centroids = None
        self.distance_computations_ = 0

    @property
    def distance_computations(self):
        return self.distance_computations_

    def _createPartitions(self, old_partitions):
        partitions = []
        for p in old_partitions:
            partitions.extend(self.binary_partition(p.data, max_=p.max, min_=p.min, thresholds=p.thresholds))
        return partitions

    def fit(self, X, y=None):
        X = np.array(X)
        self.reset()

        partitions = [Partition(X, np.full((X.shape[1],), 1), np.full((X.shape[1],), -1), np.full((X.shape[1],), 0))]
        num_partitions = 0

        while num_partitions < self.max_iter:
            # build partitions for loop
            partitions = self._createPartitions(partitions)

            # extract representatives and cardinality from partitions
            R = np.array([p.R for p in partitions])
            cardinality = np.array([p.cardinality for p in partitions])

            # initialize clusters if the number of partitions is enough for the number of clusters
            if len(partitions) < self.n_clusters:
                # if there are not enough partitions, go to the next loop iteration
                continue
            elif self.centroids is None:
                # initialize the centers
                centers = np.random.choice(range(len(cardinality)), self.n_clusters, replace=False)
                self.centroids = R[centers]
                continue
            elif len(partitions) >= X.shape[0]:
                # partitions have reached the number of examples, no point in continuing
                break

            self._cluster(R, cardinality)
            # if we reach here, it means we have initialized the centers and we are doing a kmeans.
            # old_centers = self.centroids.copy()
            num_partitions += 1

        return self

    def _cluster(self, R, cardinality):
        km = KMeans(
            n_clusters=self.n_clusters, 
            init=self.centroids,
            n_init=1,
            n_jobs=-1, 
            algorithm='full', # lloyd
        ).fit(
            R,
            sample_weight=cardinality
        )
        self.centroids = km.cluster_centers_
        self.distance_computations_ += km.n_iter_ * self.n_clusters * R.shape[0]

    @staticmethod
    def binary_partition(X, max_=None, min_=None, thresholds=None):
        n_dim = X.shape[1]

        if thresholds is None:
            thresholds = np.zeros((n_dim,))

        if max_ is None:
            max_ = np.ones((n_dim,))

        if min_ is None:
            min_ = -1*np.ones((n_dim,))

        # build binary classification array for subspace
        comps = np.array([X[:, dim] > thresholds[dim] for dim in range(n_dim)]).astype(int)

        # convert bits to uint8
        hypercube_idx = np.packbits(comps, axis=0, bitorder='little').flatten()

        partitions = []

        for i in sorted(set(hypercube_idx)):
            partition_X = X[hypercube_idx == i]
            sub_max = max_.copy()
            sub_min = min_.copy()
            sub_th = thresholds.copy()
            bin_rep = np.unpackbits(np.array([i], dtype=np.uint8), bitorder='little', count=n_dim)
            for j, b in enumerate(bin_rep):
                if b == 1:
                    sub_min[j] = sub_th[j]
                else:
                    sub_max[j] = sub_th[j]
                sub_th[j] = (sub_max[j] + sub_min[j])/2
            partitions.append(Partition(partition_X, sub_max, sub_min, sub_th))
        return partitions

    def predict(self, X):
        if self.centroids is None:
            raise ValueError("Clusters have not been initialized, call KMeans.fit(X) first")
        X = np.array(X)
        if self.centroids.shape[1] != X.shape[1]:
            raise ValueError(f"{X.shape} is an invalid X structure for kmeans of centers {self.centroids.shape}")
        return self._predictClusters(X)

    def _predictClusters(self, X):
        """
        Compute the distances from each of the data points in data to each of the centers.
        data is of shape (n_samples, n_features) and centers is of shape (n_clusters, n_features).
        It will result in a l2distances matrix of shape (n_samples, n_clusters) of which the
        argmin function will return a (n_samples,) vector with the cluster assignation.
        """
        l2distances = cdist(X, self.centroids)
        return np.argmin(l2distances, axis=1)
import numpy as np
import logging
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans

# struct
class Partition:
    def __init__(self, indexes, max, min, thresholds):
        self.indexes = indexes
        self.max = max.astype(np.float32)
        self.min = min.astype(np.float32)
        self.thresholds = thresholds.astype(np.float32)

    def __str__(self):
        return f"{self.__class__.__name__}: [min={self.min}, max={self.max}, thresholds={self.thresholds}]"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.indexes)

    def representative(self, data):
        return np.mean(data[self.indexes], axis=0)


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

    def _create_partitions(self, X, old_partitions):
        partitions = []
        for partition in old_partitions:
            partitions.extend(self.binary_partition(X, partition))
        return partitions

    def fit(self, X, y=None):
        X = np.array(X)
        n_dim = X.shape[1]
        self.reset()

        partitions = [Partition(np.arange(X.shape[0]), np.full((X.shape[1],), 1), np.full((X.shape[1],), -1), np.full((X.shape[1],), 0))]
        num_partitions = 0

        while num_partitions < self.max_iter:
            # build partitions for loop
            partitions = self._create_partitions(X, partitions)

            # extract representatives and cardinality from partitions
            R, cardinality = self._compute_partitions_meta(X, partitions, n_dim)

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

            # if we reach here, it means we have initialized the centers and we are doing a kmeans.
            self._cluster(R, cardinality)
            num_partitions += 1

        return self

    def _compute_partitions_meta(self, X, partitions, n_dim):
        R, cardinality = np.empty((len(partitions), n_dim)), np.empty((len(partitions),))
        for i, p in enumerate(partitions):
            R[i] = p.representative(X)
            cardinality[i] = len(p)
        return R, cardinality

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

    def binary_partition(self, X, partition):
        n_dim = X.shape[1]
        X = X[partition.indexes]

        # build binary classification array for subspace
        comps = X > partition.thresholds[None, :]

        # convert bits to uint8
        hypercube_idx = np.packbits(comps, axis=1, bitorder='little').flatten()

        return self._create_sub_partitions(hypercube_idx, partition.indexes, max_=partition.max, min_=partition.min, thresholds=partition.thresholds, n_dim=n_dim)

    def _create_sub_partitions(self, hypercube_idx, indexes, max_, min_, thresholds, n_dim):
        partitions = []
        for i in sorted(set(hypercube_idx)):
            partition_indexes = indexes[hypercube_idx == i]
            sub_max = max_.copy()
            sub_min = min_.copy()
            sub_th = thresholds.copy()
            bin_rep = np.unpackbits(np.array([i], dtype=np.uint8), bitorder='little', count=n_dim)
            sub_min, sub_max = self._modify_ranges(bin_rep, sub_min, sub_max, sub_th)
            sub_th = (sub_max + sub_min)/2
            partitions.append(Partition(partition_indexes, sub_max, sub_min, sub_th))
        return partitions

    def _modify_ranges(self, bin_rep, sub_min, sub_max, sub_th):
        for j, b in enumerate(bin_rep):
            if b == 1:
                sub_min[j] = sub_th[j]
            else:
                sub_max[j] = sub_th[j]
        return sub_min, sub_max

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

if __name__ == '__main__':
    data = 2 * np.random.random((int(1e6), 2)) - .5
    RPKM().fit(data)

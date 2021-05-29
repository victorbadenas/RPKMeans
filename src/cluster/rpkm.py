"""
.. module:: RPKM
RPKM
*************
:Description: RPKM
    
:Authors: victor badenas (victor.badenas@gmail.com)
    
:Version: 0.1.0
:Created on: 01/06/2021 11:00 
"""

__title__ = 'RPKM'
__version__ = '0.1.0'
__author__ = 'victor badenas'

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, ClassifierMixin
from sklearn.cluster import KMeans


class Subset:
    __doc__ = """struct containing the instance index of the subset\'s data
    as well as the max and min arrays of the data and the thresholds to be
    applied to the next partition to create from this subset."""

    def __init__(self, indexes, max, min, thresholds):
        self.indexes = indexes
        self.max = max.astype(np.float32)
        self.min = min.astype(np.float32)
        self.thresholds = thresholds.astype(np.float32)

    def __str__(self):
        """string representation of the subset

        Returns
        ----------
            str: string description of the class
        """        
        return "{}: [min={}, max={}, thresholds={}]".format(
            self.__class__.__name__,
            self.min,
            self.max,
            self.thresholds,
        )

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.indexes)

    def representative(self, data):
        """compute the representative of the partition as the mean of the
        instances.

        Parameters
        ----------
            data: np.array
                array of shape (n_samples, n_features) with the data.

        Returns:
            np.ndarray: 
                of shape (n_features,) with the representative of the subset
        """
        return np.mean(data[self.indexes], axis=0)


class RPKM(BaseEstimator, ClusterMixin, ClassifierMixin):
    __version__ = '0.1.0'
    __doc__ = """Recusive Pertition based K-Means. Finds an approximation of 
    the K-Means cluster centers by partitioning the feature space in 
    d-dimensional quadtrees and creating subsets from this partitions. 
    After this, once the subsets are created, a weighted lloyd algorithms 
    is fitted and the centers obtained are used as initialization of the 
    next step."""

    def __init__(self, n_clusters=8, max_iter=6, distance_threshold=1e-4, n_jobs=-1):
        """initializer of the rpkm class.

        Parameters
        ----------
            n_clusters : (int, optional)
                number of clusters. Defaults to 8.
            max_iter: (int, optional)
                maximum depth of the quadtree to divide the available feature space. Defaults to 6.
            distance_threshold: (float, optional)
                maximum distance accepted to stop the algorithm. Defaults to 1e-4.
            **kwargs: ([dict], optional)
                Extra arguments for the weighted lloyd algorithm computed in each partition.
        """
        self.n_clusters = n_clusters
        self.reset()
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.distance_threshold = distance_threshold

    def reset(self):
        """reset the estimator. Sets the centroids to None and resets some internal metrics.
        """
        self.centroids = None
        self.distance_computations_ = 0
        self.instance_ratio_ = -1
        self.labels_ = None
        self.n_iter_ = 0

    @property
    def distance_computations(self):
        return self.distance_computations_

    @property
    def instance_ratio(self):
        return self.instance_ratio_

    def _create_partition(self, X, old_partition):
        """creates a list with the subpartitions created with each partition in old_partitions.

        Parameters
        ----------
            X: np.ndarray
                numpy array of shape (n_samples, n_features). The dataset to fit.
            old_partitions: list
                list containing Partition objects. Last iteration's partitions.

        Returns
        ----------
            list:
                thinner partitions created from the subsets in old_partitions
        """

        partition = []
        for subset in old_partition:
            partition.extend(self.binary_partition(X, subset))
        return partition

    def fit(self, X, y=None):
        """fit the object to the dataset X.

        Parameters
        ----------
            X: np.ndarray
                numpy array of shape (n_samples, n_features). The dataset to fit.
            y: None
                not used. Used only for API consistency.

        Returns
        ----------
            PRKM:
                fitted object
        """
        # copy dataset into memory and unpack shapes
        X = self._validate_data(X, y=None, ensure_2d=True)
        n_samples = X.shape[0]

        # reset estimator
        self.reset()

        # initialize partitions list as a one partition of the whole dataset.
        partition = [Subset(np.arange(n_samples), np.full((X.shape[1],), 1), np.full((X.shape[1],), -1), np.full((X.shape[1],), 0))]

        # loop until the max number of iterations are met
        num_partition = 0
        while num_partition < self.max_iter:
            # build partition for loop
            partition = self._create_partition(X, partition)

            # extract representatives and cardinality from partitions
            R, cardinality = self._compute_partition_meta(X, partition)

            # initialize clusters if the number of partitions is enough for the number of clusters
            if len(partition) < self.n_clusters:
                # if there are not enough partitions, go to the next loop iteration
                continue

            elif self.centroids is None:
                # initialize the centers if there are more partitions than centers 
                # and the centers have not yet been initialized
                centers = np.random.choice(range(R.shape[0]), self.n_clusters, replace=False)
                self.centroids = R[centers]

            elif len(partition) >= n_samples:
                # partitions have reached the number of examples, no point in continuing
                break

            # if we reach here, it means we have initialized the centers and will fit a wl.
            self._cluster(R, cardinality)
            num_partition += 1

        # store instance ratio for last partition
        self.instance_ratio_ = len(partition) / n_samples
        self.labels_ = self.predict(X)
        self.n_iter_ = num_partition
        return self

    def _compute_partition_meta(self, X, subsets):
        """compute set of representatives and cardinality arrays for a given list of subsets.

        Parameters
        ----------
            X: np.array
                numpy array of shape (n_samples, n_features). The dataset to fit.
            subsets: list
                list of subsets of length L.

        Returns
        ----------
            np.array:
                set of representatives of shape (L, n_features)
            np.array:
                cardinalities of shape (L,)
        """

        # initialize vectors 
        R, cardinality = np.empty((len(subsets), self.n_features_in_)), np.empty((len(subsets),))

        # extract representative and cardinality arrays for each partition.
        for i, p in enumerate(subsets):
            R[i] = p.representative(X)
            cardinality[i] = len(p)

        # return
        return R, cardinality

    def _cluster(self, R, cardinality):
        """use the previous iteration\'s centers as an initialization for
        a weighted lloyd algorithm and fit the wl to the representatives and 
        cardinality of the subsets of the partition. Extract centers from the
        WL algorithm\'s last iteration.

        Parameters
        ----------
            R: np.ndarray
                array of shape (n_subsets, n_features)
            cardinality: np.ndarray
                array of shape (n_subsets,)
        """

        km = KMeans(
            n_clusters=self.n_clusters, 
            init=self.centroids,
            n_init=1,
            algorithm='full', # lloyd
            n_jobs=self.n_jobs
        ).fit(
            R,
            sample_weight=cardinality
        )
        self.centroids = km.cluster_centers_
        self.distance_computations_ += km.n_iter_ * self.n_clusters * R.shape[0]

    def binary_partition(self, X, subset):
        """create binary partition for the current subset.

        Parameters
        ----------
            X: np.array
                numpy array of shape (n_samples, n_features). The dataset to fit.
            subset: Subset
                class containing the indexes, range and threshold information for a
                subset.

        Returns
        ----------
            list:
                subsets for the partition
        """
        X = X[subset.indexes]

        # build binary classification array for subspace
        comps = X > subset.thresholds[None, :]

        # convert bits to int
        hypercube_idx = bintoint(comps, comps.shape[-1])

        return self._create_sub_partitions(hypercube_idx, subset.indexes, max_=subset.max, min_=subset.min, thresholds=subset.thresholds)

    def _create_sub_partitions(self, hypercube_idx, indexes, max_, min_, thresholds):
        """given a set of hypercube indexes, create a list of partitions with the indexes of the
        instances assigned to each partition.

        Parameters
        ----------
            hypercube_idx: iterable
                int representation of the hypercube subsets.
            indexes: 
                indexes of the subset\'s instances.
            max_: np.ndarray
                np.array of shape (n_features, ) containing the min for each dimension
            min_: np.ndarray
                np.array of shape (n_features, ) containing the max for each dimension
            thresholds: np.ndarray
                np.array of shape (n_features, ) containing the threshold for each dimension

        Returns
        ----------
            list:
                subsets for the partition
        """

        partition = []
        for i in sorted(set(hypercube_idx)):
            partition_indexes = indexes[hypercube_idx == i]
            sub_max = max_.copy()
            sub_min = min_.copy()
            sub_th = thresholds.copy()
            bin_rep = inttobin(i, self.n_features_in_).flatten()
            sub_min, sub_max = self._modify_ranges(bin_rep, sub_min, sub_max, sub_th)
            sub_th = (sub_max + sub_min)/2
            partition.append(Subset(partition_indexes, sub_max, sub_min, sub_th))
        return partition

    def _modify_ranges(self, bin_rep, sub_min, sub_max, sub_th):
        """shift range of the subset according to the binary representation of the subset.
        If a bit is 1 means that the comparison was True and thus the range must be shifted up.
        This is done by assigining the current threshold value to the min of the range. Else 
        the threshold value is assigned to the maximum of the range.

        Parameters
        ----------
            bin_rep: np.array
                np array of int representation 
            sub_min: np.array
                np.array of shape (n_features, ) containing the min for each dimension
            sub_max: np.array
                np.array of shape (n_features, ) containing the max for each dimension
            sub_th: np.array
                np.array of shape (n_features, ) containing the threshold for each dimension

        Returns
        ----------
            np.array:
                modified min array
            np.array:
                modified max array
        """
        sub_min[bin_rep] = sub_th[bin_rep]
        sub_max[~bin_rep] = sub_th[~bin_rep]
        return sub_min, sub_max

    def predict(self, X):
        """assign cluster indexes to a set of instances.

        Parameters
        ----------
            X: np.ndarray: 
                array of shape (n_samples, n_features). instances to be predicted.

        Raises
        ----------
            ValueError: if the clusters have not yet been initialized
            ValueError: if the number of dimensions between centers and n_features
                is not the same.

        Returns
        ----------
            np.ndarray:
                of shape (n_samples,) with an integer representation of the cluster 
                assignation
        """

        #check if fitted
        if self.centroids is None:
            raise ValueError("Clusters have not been initialized, call KMeans.fit(X) first")

        # copy the X dataset and convert to np.array if another type is passed.
        X = np.array(X)

        # check if n_features is number of dimensions in the centroids
        if self.centroids.shape[1] != X.shape[1]:
            raise ValueError(f"{X.shape} is an invalid X structure for kmeans of centers {self.centroids.shape}")

        # predict the clusters
        return self._predictClusters(X)

    def _predictClusters(self, X):
        """Compute the distances from each of the data points in data to each of the centers.
        The assignation is done by taking the index of the cluster nearest to each instance.

        Parameters
        ----------
            X: np.ndarray:
                array of shape (n_samples, n_features) with the point to predict

        Returns
        ----------
            np.ndarray:
                cluster asignation
        """

        l2distances = cdist(X, self.centroids)
        return np.argmin(l2distances, axis=1)

def inttobin(value, n_dim):
    return (((value & (1 << np.arange(n_dim)))) > 0)

def bintoint(value, n_dim):
    return (value*2**np.arange(n_dim)).sum(axis=1)

import numpy.ma as ma
import numpy as np
from sklearn import datasets
from statistics import NormalDist
from scipy.spatial.distance import cdist


class ArtificialDatasetGenerator:
    def __init__(self, 
        n_centers: int,
        n_features: int,
        n_samples: int,
        normalize: bool=True,
        n_replicas: int=-1,
        **dataset_kwargs):

        self.n_centers_ = n_centers
        self.n_features_ = n_features
        self.n_samples_ = n_samples
        self.normalize = normalize
        self.n_replicas_ = n_replicas
        self.dataset_kwargs = dataset_kwargs
        self.max_overlap_ = .05

    def __call__(self):
        x, y = self._generate_dataset()
        if self.normalize:
            x = self._normalize_dataset(x)
        return x, y

    def __getitem__(self, idx):
        if self.n_replicas_ > 0:
            if idx >= self.n_replicas_:
                raise StopIteration
        return self()

    @property
    def n_centers(self):
        return self.n_centers_

    @property
    def n_features(self):
        return self.n_features_

    @property
    def n_samples(self):
        return self.n_samples_

    @n_samples.setter
    def n_samples(self, value:int):
        if n_samples > 0:
            self.n_samples_ = value
        raise ValueError(f'n_samples must be > 0, not {value}')

    @property
    def shape(self):
        return ((self.n_samples_, self.n_features_), (self.n_samples_,))

    def __str__(self):
        return str(self.__dict__())

    def __repr__(self):
        return self.__str__()

    def _find_std(self, centers):
        centerdist = cdist(centers, centers)
        min_val = ma.masked_array(centerdist, mask=centerdist==0).min()
        center_indexes = np.where(centerdist == min_val)[0]
        closest_centers = centers[center_indexes]
        sigma = 0.1
        overlap = NormalDist(mu=0, sigma=sigma).overlap(NormalDist(mu=min_val, sigma=sigma))
        while not np.isclose(overlap, self.max_overlap_):
            if overlap < self.max_overlap_:
                sigma *= 1 + (self.max_overlap_ - overlap)
            else:
                sigma /= 1 + (overlap - self.max_overlap_)
            overlap = NormalDist(mu=0, sigma=sigma).overlap(NormalDist(mu=min_val, sigma=sigma))
        return sigma

    def _generate_dataset(self):
        centers = 2*(np.random.random((self.n_centers_, self.n_features))-.5)
        sigma = self._find_std(centers)
        return datasets.make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=centers,
            center_box=(-1, 1),
            cluster_std=sigma,
            **self.dataset_kwargs)

    def _normalize_dataset(self, x):
        return x / np.abs(x).max(axis=0, keepdims=True)

if __name__ == '__main__':
    x, y = ArtificialDatasetGenerator(4, 2, 100)()
import numpy as np
from sklearn import datasets

class ArtificialDatasetGenerator:
    def __init__(self, K, d, n):
        self.K = K
        self.d = d
        self.n = n

    def __call__(self):
        return self._generate_dataset()

    def __iter__(self):
        return self

    def __getitem__(self, *args, **kwargs):
        return self._generate_dataset()

    @property
    def shape(self):
        return ((self.n, self.d), (self.n,))

    def __str__(self):
        return str(self.__dict__())

    def __repr__(self):
        return self.__str__()

    def _generate_dataset(self):
        return datasets.make_blobs(n_samples=self.n, n_features=self.d, centers=self.K)

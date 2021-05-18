import numpy as np
from sklearn import datasets


class ArtificialDatasetGenerator:
    def __init__(self, 
        n_centers: int,
        n_features: int,
        n_samples: int,
        normalize: bool=True,
        **dataset_kwargs):

        self.n_centers_ = n_centers
        self.n_features_ = n_features
        self.n_samples_ = n_samples
        self.normalize = normalize
        self.dataset_kwargs = dataset_kwargs

    def __call__(self):
        x, y = self._generate_dataset()
        if self.normalize:
            x = self._normalize_dataset(x)
        return x, y

    def __iter__(self):
        return self

    def __getitem__(self, *args, **kwargs):
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

    def _generate_dataset(self):
        return datasets.make_blobs(
            n_samples=self.n_samples,
            n_features=self.n_features,
            centers=self.n_centers_,
            center_box=(-1, 1),
            **self.dataset_kwargs)

    def _normalize_dataset(self, x):
        return x / np.abs(x).max(axis=0, keepdims=True)

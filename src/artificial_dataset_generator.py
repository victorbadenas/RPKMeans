import numpy as np

class ArtificialDatasetGenerator:
    def __init__(self, K, d, n):
        self.K = K
        self.d = d
        self.n = n

    def __call__(self):
        return self._generate_dataset()

    @property
    def shape(self):
        return (self.n, self.d)

    def __str__(self):
        return str(self.__dict__())

    def __repr__(self):
        return self.__str__()

    def _generate_dataset(self):
        pass
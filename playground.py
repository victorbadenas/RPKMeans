import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    afg = ArtificialDatasetGenerator(K=4, d=2, n=100)
    dataset = afg()

    plt.scatter()



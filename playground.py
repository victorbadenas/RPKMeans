import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_classification

class ArtificialDatasetGenerator:
    def __init__(self, K, d, n, var=0.05, normalize=False):
        self.K = K
        self.d = d
        self.n = n
        self.var = var
        self.normalize = normalize

    def __call__(self):
        return self._generate_dataset()

    @property
    def shape(self):
        return (self.n, self.d)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def _generate_dataset(self):
        # dataset = None
        # labels = None
        centers = self._generate_centers()
        # for k, center in enumerate(centers):
        #     cluster_data = np.random.multivariate_normal(np.zeros(self.d), self.var*np.eye(self.d), size=self.n//self.K)
        #     cluster_data += center
        #     cluster_labels = np.full(self.n//self.K, k)
        #     if dataset is None:
        #         dataset, labels = cluster_data, cluster_labels
        #     else:
        #         dataset = np.concatenate((dataset, cluster_data), axis=0)
        #         labels = np.concatenate((labels, cluster_labels), axis=0)
        dataset, labels = make_classification(
            n_samples=self.n,
            n_features=self.d,
            n_clusters_per_class=1,
            n_classes=self.K,
            n_informative=self.d,
            n_redundant=0,
            n_repeated=0,
            class_sep=10, 
            shuffle=False
        )

        if self.normalize:
            if callable(self.normalize):
                dataset = self.normalize(dataset)
            else:
                dataset /= np.abs(dataset).max(axis=0)
        return dataset, labels

    def _generate_centers(self):
        return np.array([ [np.cos((2*i/self.K + 0.25)*np.pi), np.sin((2*i/self.K + 0.25)*np.pi)] for i in range(self.K) ])

if __name__ == '__main__':
    afg = ArtificialDatasetGenerator(K=4, d=2, n=100000, normalize=True)
    dataset, labels = afg()

    unique_labels = np.unique(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        sub_dataset = dataset[labels == label]
        plt.scatter(sub_dataset[:, 0], sub_dataset[:, 1], color=color, label=label)
    plt.legend()
    plt.show()

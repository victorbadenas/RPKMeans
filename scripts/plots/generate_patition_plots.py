# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import os
import time
import random
import warnings
from pprint import pprint

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)

colors = list(mcolors.TABLEAU_COLORS.values())[:8]
sys.path.append('../../src')

def warn(*args, **kwargs):
    pass
warnings.warn = warn

OUT_PATH = Path(f'../../images/{SEED}/')
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

from artificial_dataset_generator import ArtificialDatasetGenerator
from dataset import PandasUnsupervised
from sklearn.cluster import KMeans


# %%
def ideal_centers_gauss(x, y):
    return np.array([x[y == label].mean(axis=0) for label in set(y)])


# %%
def plot(x, y, grid_lines=None, title:str="", ideal:bool=True, alpha:float=0, range=(-1, 1), legend=True, save_path=None, show=True, **kwargs):
    plt.figure(**kwargs)
    if title != "":
        plt.title(title)
    
    if grid_lines is not None:
        if hasattr(grid_lines, '__iter__'):
            plt.hlines(grid_lines, -1, 1, colors=[alpha, alpha, alpha], linestyles='dashdot', linewidth=.3)
            plt.vlines(grid_lines, -1, 1, colors=[alpha, alpha, alpha], linestyles='dashdot', linewidth=.3)
    else:
        plt.grid('on')

    for i, label in enumerate(set(y)):
        subdata = x[y == label]
        plt.scatter(subdata[:,0], subdata[:,1], marker='.', label=label, color=colors[i%len(colors)], linewidths=.1)
    
    if ideal:
        centers = ideal_centers_gauss(x, y)
        plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k')

    if range is not None:
        plt.xlim(range)
        plt.ylim(range)

    if legend:
        plt.legend()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


# %%
def plot_ax(ax, x, y, grid_lines=None, title:str="", ideal:bool=True, alpha:float=0, range=(-1, 1), legend=True, **kwargs):
    if title != "":
        ax.set_title(title)
    
    if grid_lines is not None:
        if hasattr(grid_lines, '__iter__'):
            ax.hlines(grid_lines, -1, 1, colors=[alpha, alpha, alpha], linestyles='dashdot', linewidth=.3)
            ax.vlines(grid_lines, -1, 1, colors=[alpha, alpha, alpha], linestyles='dashdot', linewidth=.3)
    else:
        ax.grid('on')

    for i, label in enumerate(set(y)):
        subdata = x[y == label]
        ax.scatter(subdata[:,0], subdata[:,1], marker='.', label=label, color=colors[i%len(colors)])
    
    if ideal:
        centers = ideal_centers_gauss(x, y)
        ax.scatter(centers[:, 0], centers[:, 1], marker='x', color='k')

    if range is not None:
        ax.set_xlim(range)
        ax.set_ylim(range)

    if legend:
        ax.legend()
    return ax


# %%
def plot_center_lines(centers, data=None, alpha=.3, title:str="", save_path=None, show=True, range=[-1, 1], **kwargs):
    plt.figure(**kwargs)
    legend_elements = [Line2D([0], [0], color='k', marker='x', label='centers')]

    if data is not None:
        plt.scatter(data[:,0], data[:,1], marker='.', color=[alpha, alpha, alpha])
        legend_elements.append(Line2D([0], [0], color=[alpha, alpha, alpha], marker='.', label='dataset'))

    for i, idx_center in enumerate(centers.swapaxes(0, 1)):
        plt.scatter(idx_center[:, 0], idx_center[:, 1], marker='x', color=colors[i%len(colors)])
        plt.plot(idx_center[:, 0], idx_center[:, 1], color=colors[i%len(colors)], )

    if title != "":
        plt.title(title)

    plt.legend(handles=legend_elements)
    plt.grid('on')

    if range is not None:
        plt.xlim(range)
        plt.ylim(range)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


# %%
def plot_center_lines_iteration_ax(ax, old_centers, new_centers, representatives=None, data=None, alpha=.2, title:str="", range=[-1, 1], **kwargs):
    # plt.figure(**kwargs)
    legend_elements = [Line2D([0], [0], color='k', marker='x', label='old_centers'),
                       Line2D([0], [0], color='k', marker='o', label='new_centers')]

    if data is not None:
        ax.scatter(data[:,0], data[:,1], marker='.', color=[alpha, alpha, alpha])
        legend_elements.append(Line2D([0], [0], color=[alpha, alpha, alpha], marker='.', label='dataset'))

    if representatives is not None:
        ax.scatter(representatives[:,0], representatives[:,1], marker='.', color='k')
        legend_elements.append(Line2D([0], [0], color='k', marker='.', label='representatives'))

    for i, (oc, nc) in enumerate(zip(old_centers, new_centers)):
        ax.scatter(*oc, marker='x', color=colors[i%len(colors)])
        ax.scatter(*nc, marker='o', color=colors[i%len(colors)])
        line = np.array([oc, nc])
        ax.plot(line[:,0],line[:,1], color=colors[i%len(colors)])

    if title != "":
        ax.set_title(title)

    if range is not None:
        ax.set_xlim(range)
        ax.set_ylim(range)

    ax.legend(handles=legend_elements)
    ax.grid('on')
    return ax


# %%
n_centers = 3
n_features = 2
n_samples = 4000


# %%
# pdataset = PandasUnsupervised('../data/gas_sensor/ethylene_methane.csv').data
# pdataset = pdataset.drop(['Time_(seconds)','Methane_conc_(ppm)','Ethylene_conc_(ppm)'], axis=1)
# print(pdataset.describe())
data, labels = ArtificialDatasetGenerator(
    n_centers=n_centers,
    n_features=n_features,
    n_samples=int(n_samples),
    normalize=True,
    n_replicas=1)()


# %%
# data = pdataset.sample(n_samples, replace=False).sample(n_features, axis=1)
# data.describe()
# data = data.to_numpy()


# %%
max_iter = 6
n_clusters = n_centers
distance_threshold = 1e-3

data_indexes = np.arange(data.shape[0])
n_dim = data.shape[1]


# %%
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


# %%
def _modify_ranges(bin_rep, sub_min, sub_max, sub_th):
    for j, b in enumerate(bin_rep):
        if b == 1:
            sub_min[j] = sub_th[j]
        else:
            sub_max[j] = sub_th[j]
    return sub_min, sub_max


# %%
def _create_sub_partitions(hypercube_idx, indexes, max_, min_, thresholds, n_dim):
    partitions = []
    for i in sorted(set(hypercube_idx)):
        partition_indexes = indexes[hypercube_idx == i]
        sub_max = max_.copy()
        sub_min = min_.copy()
        sub_th = thresholds.copy()
        bin_rep = np.unpackbits(np.array([i], dtype=np.uint8), bitorder='little', count=n_dim)
        sub_min, sub_max = _modify_ranges(bin_rep, sub_min, sub_max, sub_th)
        sub_th = (sub_max + sub_min)/2
        partitions.append(Partition(partition_indexes, sub_max, sub_min, sub_th))
    return partitions


# %%
def binary_partition(X, partition):
    n_dim = X.shape[1]
    X = X[partition.indexes]

    # build binary classification array for subspace
    comps = X > partition.thresholds[None, :]

    # convert bits to uint8
    hypercube_idx = np.packbits(comps, axis=1, bitorder='little').flatten()

    return _create_sub_partitions(hypercube_idx, partition.indexes, max_=partition.max, min_=partition.min, thresholds=partition.thresholds, n_dim=n_dim)


# %%
def _create_partitions(X, old_partitions):
    partitions = []
    for partition in old_partitions:
        partitions.extend(binary_partition(X, partition))
    return partitions


# %%
def _compute_partitions_meta(X, partitions, n_dim):
    R, cardinality = np.empty((len(partitions), n_dim)), np.empty((len(partitions),))
    for i, p in enumerate(partitions):
        R[i] = p.representative(X)
        cardinality[i] = len(p)
    return R, cardinality


# %%
f, axes = plt.subplots(2, 3, figsize=(25, 15))
axes = axes.flatten()

X = data
n_dim = X.shape[1]
centroids = None
distance_computations_ = 0

partitions = [Partition(np.arange(X.shape[0]), np.full((X.shape[1],), 1), np.full((X.shape[1],), -1), np.full((X.shape[1],), 0))]
num_partitions = 0

accum_centers = []
init_iteration = 0
all_partitions = []

while num_partitions < max_iter:
    # build partitions for loop
    partitions = _create_partitions(X, partitions)

    # extract representatives and cardinality from partitions
    R, cardinality = _compute_partitions_meta(X, partitions, n_dim)

    # initialize clusters if the number of partitions is enough for the number of clusters
    if len(partitions) < n_clusters:
        # if there are not enough partitions, go to the next loop iteration
        init_iteration += 1
        continue

    elif centroids is None:
        # initialize the centers
        centers = np.random.choice(range(len(cardinality)), n_clusters, replace=False)
        centroids = R[centers]
        accum_centers.append(centroids)
        

    elif len(partitions) >= X.shape[0]:
        # partitions have reached the number of examples, no point in continuing
        break

    old_centers = centroids.copy()
    km = KMeans(
        n_clusters=n_clusters, 
        init=centroids,
        n_init=1,
        n_jobs=-1, 
        algorithm='full', # lloyd
    ).fit(
        R,
        sample_weight=cardinality
    )
    centroids = km.cluster_centers_
    distance_computations_ += km.n_iter_ * n_clusters * R.shape[0]

    accum_centers.append(centroids)
    all_partitions.append((num_partitions+init_iteration, partitions))

    plot_center_lines_iteration_ax(axes[num_partitions], old_centers, centroids, representatives=R, data=X, alpha=.5, title=f"partition {num_partitions}", range=[-1, 1])

    num_partitions += 1
plt.savefig(OUT_PATH/'steps.png')
plt.show()


# %%
f, axes = plt.subplots(2, 3, figsize=(25, 15))
axes = axes.flatten()

for j, (i, partitions) in enumerate(all_partitions):
    thresholds = np.linspace(-1, 1, 2**(i+1)+1)
    plot_data = [X[p.indexes] for p in partitions]
    labels = [np.full((len(p),), j) for j, p in enumerate(partitions)]
    plot_ax(axes[j], np.concatenate(plot_data), np.concatenate(labels), grid_lines=thresholds, ideal=False, title=f'depth {i}', figsize=(10, 10), legend=False)
plt.savefig(OUT_PATH/'alldepths.png')


# %%
plot_center_lines(np.array(accum_centers), data, figsize=(9, 9), save_path=OUT_PATH/'summary_iteration.png', range=[-1, 1], alpha=.7)


# %%




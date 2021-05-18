import sys
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import os
import time
import random
from pprint import pprint
random.seed(1995)
np.random.seed(1995)

colors = list(mcolors.TABLEAU_COLORS.values())[:8]
sys.path.append('../src')

if not os.path.exists('../images'):
    os.makedirs('../images/')

from artificial_dataset_generator import ArtificialDatasetGenerator
from clustering.kmeans import KMeans


n_centers = 4
n_features = 2
n_samples = 10000


adg = ArtificialDatasetGenerator(n_centers=n_centers, n_features=n_features, n_samples=n_samples, normalize=True, cluster_std=.1)


x, y = adg()


def ideal_centers_gauss(x, y):
    return np.array([x[y == label].mean(axis=0) for label in set(y)])


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


plot(x, y, title='Ground Truth', figsize=(6, 6), save_path='../images/ground_truth_random.png', )


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
        ax.scatter(*nc, marker='x', color=colors[i%len(colors)])
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


max_iter = 6
n_clusters = n_centers
distance_threshold = 1e-3

data = x
data_indexes = np.arange(data.shape[0])
n_dim = data.shape[1]


from collections import defaultdict
def binary_partition(data, depth=0, max_depth=2, max_=None, min_=None, thresholds=None, partitions=None):
    n_dim = data.shape[1]

    if thresholds is None:
        thresholds = np.zeros((n_dim,))

    if max_ is None:
        max_ = np.ones((n_dim,))
    elif isinstance(max_, int):
        max_ = max_ * np.ones((n_dim,))

    if min_ is None:
        min_ = -1*np.ones((n_dim,))
    elif isinstance(min_, int):
        min_ = min_ * np.ones((n_dim,))

    if depth > max_depth - 1:
        return

    if partitions is None:
        partitions = dict()

    # build binary classification array for subspace
    comps = np.array([data[:, dim] > thresholds[dim] for dim in range(n_dim)]).astype(int)

    # convert bits to uint8
    hypercube_idx = np.packbits(comps, axis=0, bitorder='little').flatten()

    local_partitions = [data[hypercube_idx == i] for i in set(hypercube_idx)]
    if depth not in partitions:
        partitions[depth] = list()
    partitions[depth].extend(local_partitions)

    for i, partition in zip(set(hypercube_idx), local_partitions):
        sub_max = max_.copy()
        sub_min = min_.copy()
        sub_th = thresholds.copy()
        bin_rep = np.unpackbits(np.array([i], dtype=np.uint8), bitorder='little', count=n_dim)
        for i, b in enumerate(bin_rep):
            if b == 1:
                sub_min[i] = sub_th[i]
            else:
                sub_max[i] = sub_th[i]
            sub_th[i] = (sub_max[i] + sub_min[i])/2
        ret = binary_partition(partition, 
                            depth=depth+1, 
                            max_depth=max_depth,
                            max_=sub_max,
                            min_=sub_min,
                            thresholds=sub_th,
                            partitions=partitions)
        if isinstance(ret, dict):
            partitions = ret
    return partitions

partitions = binary_partition(x, max_depth=max_iter)


partition_stats = dict()
for k, v in partitions.items():
    partition_stats[k] = (
        np.array([i.mean(axis=0) for i in v]),
        [float(i.shape[0]) for i in v]
    )


f, axes = plt.subplots(2, 3, figsize=(25, 15))
axes = axes.flatten()

accum_centers = []
centers = None
for i, (R, cardinality) in partition_stats.items():
    if len(cardinality) < n_clusters:
        continue
    elif centers is None:
        centers = np.random.choice(range(len(cardinality)), n_clusters, replace=False)
        centers = R[centers]
        accum_centers.append(centers)
        if centers.shape[0] == n_clusters:
            continue

    old_centers = centers.copy()
    centers = KMeans(n_clusters=n_centers, init=centers, n_init=1, verbose=True).fit(R, sample_weights=cardinality).centers
    accum_centers.append(centers)
    plot_center_lines_iteration_ax(axes[i], old_centers, centers, representatives=R, data=data, alpha=.3, title=f"partition {i}", range=[-1, 1])

plt.savefig('../images/steps.png')
plt.show()


f, axes = plt.subplots(2, 3, figsize=(25, 15))
axes = axes.flatten()

for k, v in partitions.items():
    thresholds = np.linspace(-1, 1, 2**(k+1)+1)
    labels = [np.full((mat.shape[0],), i) for i, mat in enumerate(v)]
    plot_ax(axes[k], np.concatenate(v), np.concatenate(labels), grid_lines=thresholds, ideal=False, title=f'depth {k}', figsize=(10, 10), legend=False)
plt.savefig('../images/alldepths.png')


plot_center_lines(np.array(accum_centers), data, figsize=(9, 9), save_path='../images/summary_iteration.png', range=[-1, 1])



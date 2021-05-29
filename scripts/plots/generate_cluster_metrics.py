# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

INPUT_PATH = Path('results/quality_artificial_agg.csv')

RESULTS_FOLDER = Path('images/')
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

plot_data = pd.read_csv(INPUT_PATH, sep=',')

model_kwargs = ['algorithm', 'param']
n_clusters_unique = plot_data.n_clusters.unique()
n_dims_unique = plot_data.n_dims.unique()
n_samples_unique = plot_data.n_samples.unique()
n_clusters_unique, n_dims_unique, n_samples_unique

f, ax = plt.subplots(len(n_clusters_unique)*len(n_dims_unique), len(n_samples_unique), sharex=True, figsize=(30, 25))
for i, (n_clusters, n_dims) in enumerate(product(n_clusters_unique, n_dims_unique)):
    subplot_data = plot_data[(plot_data.n_clusters == n_clusters) & (plot_data.n_dims == n_dims)]
    for j, n_samples in enumerate(n_samples_unique):
        bar_data = subplot_data[subplot_data.n_samples == n_samples]
        model_names = sorted(set(zip(*[bar_data[kw] for kw in model_kwargs])))
        model_names = [f"{model_name[0]}:{model_name[1]}" for model_name in model_names]
        y_data = bar_data['silhouette']
        ax[i][j].bar(range(len(model_names)), y_data, tick_label=model_names)
        ax[i][j].tick_params(axis='x', rotation=90)
        ax[i][j].grid('on')
        ax[i][j].set_title(f"n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}")
plt.savefig(RESULTS_FOLDER / 'silouette_score.png')

f, ax = plt.subplots(len(n_clusters_unique)*len(n_dims_unique), len(n_samples_unique), sharex=True, figsize=(30, 25))
for i, (n_clusters, n_dims) in enumerate(product(n_clusters_unique, n_dims_unique)):
    subplot_data = plot_data[(plot_data.n_clusters == n_clusters) & (plot_data.n_dims == n_dims)]
    for j, n_samples in enumerate(n_samples_unique):
        bar_data = subplot_data[subplot_data.n_samples == n_samples]
        model_names = sorted(set(zip(*[bar_data[kw] for kw in model_kwargs])))
        model_names = [f"{model_name[0]}:{model_name[1]}" for model_name in model_names]
        y_data = bar_data['calinski_harabasz']
        ax[i][j].bar(range(len(model_names)), y_data, tick_label=model_names)
        ax[i][j].tick_params(axis='x', rotation=90)
        ax[i][j].grid('on')
        ax[i][j].set_title(f"n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}")
plt.savefig(RESULTS_FOLDER / 'calinski_harabasz_score.png')

f, ax = plt.subplots(len(n_clusters_unique)*len(n_dims_unique), len(n_samples_unique), sharex=True, figsize=(30, 25))
for i, (n_clusters, n_dims) in enumerate(product(n_clusters_unique, n_dims_unique)):
    subplot_data = plot_data[(plot_data.n_clusters == n_clusters) & (plot_data.n_dims == n_dims)]
    for j, n_samples in enumerate(n_samples_unique):
        bar_data = subplot_data[subplot_data.n_samples == n_samples]
        model_names = sorted(set(zip(*[bar_data[kw] for kw in model_kwargs])))
        model_names = [f"{model_name[0]}:{model_name[1]}" for model_name in model_names]
        y_data = bar_data['davies_bouldin']
        ax[i][j].bar(range(len(model_names)), y_data, tick_label=model_names)
        ax[i][j].tick_params(axis='x', rotation=90)
        ax[i][j].grid('on')
        ax[i][j].set_title(f"n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}")
plt.savefig(RESULTS_FOLDER / 'davies_bouldin_score.png')

kmeans_data = plot_data[plot_data.algorithm == 'k-means++']
rpkm_data = plot_data[plot_data.algorithm == 'rpkm']

import numpy as np
results = np.empty((3, 6))
for i, metric in enumerate(list(rpkm_data.columns)[-3:]):
    for j, param in enumerate(rpkm_data.param.unique()):
        subdata = rpkm_data[rpkm_data.param == param]
        a = subdata[metric].to_numpy()
        b = kmeans_data[metric].to_numpy()
        if metric == 'silhouette':
            a = 0.5*(a+1) # map from [-1, 1] to [0,1]
            b = 0.5*(b+1) # map from [-1, 1] to [0,1]
        score_relative = np.abs(a-b) / (a+b)
        results[i,j] = score_relative.mean()
pd.DataFrame(
    np.array(results), 
    index=list(rpkm_data.columns)[-3:], 
    columns=rpkm_data.param.unique()
).to_csv(RESULTS_FOLDER / 'relative_cluster_metrics_wrt_kmeanspp.csv')


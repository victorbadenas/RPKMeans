import logging
import os
import random
import argparse
import sys
import time
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

sys.path.append('src')
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from artificial_dataset_generator import ArtificialDatasetGenerator
from cluster import RPKM
from sklearn.cluster import KMeans, MiniBatchKMeans

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)
pb = None

OUT_FOLDER = Path('results/')
N_REPLICAS = 10
N_CLUSTERS = [3, 9]
N_DIMS = [2, 4, 6, 8]
N_SAMPLES = [1e2, 1e3, 1e4, 1e5, 1e6]

KWARGS = [
    {"algorithm": 'rpkm', "param": 1},
    {"algorithm": 'rpkm', "param": 2},
    {"algorithm": 'rpkm', "param": 3},
    {"algorithm": 'rpkm', "param": 4},
    {"algorithm": 'rpkm', "param": 5},
    {"algorithm": 'rpkm', "param": 6},
]

METRICS = {
    "adjusted_mutual_info": adjusted_mutual_info_score,
    "adjusted_rand": adjusted_rand_score,
    "normalized_mutual_info": normalized_mutual_info_score,
}

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def set_logger(log_file_path, debug=False):
    log_file_path = Path(log_file_path)
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(module)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    if debug:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(logging_format))
        logging.getLogger().addHandler(consoleHandler)

def compute(n_clusters, n_dims, n_samples, k_means_kwargs, computationIdx):
    logging.info(f'now computing: n_clusters={n_clusters}, n_dims={n_dims}, n_samples={n_samples}, k_means_kwargs={k_means_kwargs}')
    np.random.seed(computationIdx)
    random.seed(computationIdx)

    dataset_generator = ArtificialDatasetGenerator(
        n_centers=n_clusters,
        n_features=n_dims,
        n_samples=int(n_samples),
        normalize=True,
        n_replicas=1)

    data, labels = dataset_generator()

    clf = RPKM(
        n_clusters=n_clusters,
        max_iter=k_means_kwargs['param'],
        n_jobs=-1
    )
    pred_labels = clf.fit_predict(data)

    ref_clf = KMeans(n_clusters=n_clusters, n_init=1, init=clf.centroids, n_jobs=-1, algorithm='full')
    ref_labels = ref_clf.fit_predict(data)

    metrics = dict.fromkeys(METRICS, None)

    for metric_name in metrics:
        metrics[metric_name] = METRICS[metric_name](ref_labels, pred_labels)

    return n_clusters, n_dims, n_samples, *k_means_kwargs.values(), *metrics.values()


if __name__ == '__main__':
    set_logger('logs/cluster_equivalency_artificial.log', debug=False)

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES, KWARGS, list(range(N_REPLICAS))))

    results = [compute(*args) for args in tqdm(combinations)]
    results.sort()

    columns = ['n_clusters','n_dims','n_samples'] + list(KWARGS[0].keys()) + list(METRICS.keys())
    results = pd.DataFrame(results, columns=columns)
    results = results.fillna('None')
    results = results.groupby(columns[:5]).mean().reset_index()
    results.to_csv(OUT_FOLDER / 'cluster_equivalency_artificial_agg.csv', sep=',', index=False)

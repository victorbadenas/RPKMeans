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
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
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
N_DIMS = [2, 4, 8]
N_SAMPLES = [1e2, 1e3, 1e4]
N_THREADS = 1 # int or None to use all of them

KWARGS = [
    {"algorithm": 'k-means++', "param": None},
    {"algorithm": 'rpkm', "param": 1},
    {"algorithm": 'rpkm', "param": 2},
    {"algorithm": 'rpkm', "param": 3},
    {"algorithm": 'rpkm', "param": 4},
    {"algorithm": 'rpkm', "param": 5},
    {"algorithm": 'rpkm', "param": 6},
    {"algorithm": "mb-kmeans", "param": 100},
    {"algorithm": "mb-kmeans", "param": 500},
    {"algorithm": "mb-kmeans", "param": 1000},
]

METRICS = {
    "silhouette": silhouette_score,
    "calinski_harabasz": calinski_harabasz_score,
    "davies_bouldin": davies_bouldin_score,
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

def compute(n_clusters, n_dims, n_samples, k_means_kwargs, computationidx):
    logging.info(f'now computing: n_clusters={n_clusters}, n_dims={n_dims}, n_samples={n_samples}, k_means_kwargs={k_means_kwargs}, computationidx={computationidx}')
    np.random.seed(computationidx)
    random.seed(computationidx)

    dataset_generator = ArtificialDatasetGenerator(
        n_centers=n_clusters,
        n_features=n_dims,
        n_samples=int(n_samples),
        normalize=True,
        n_replicas=1)

    data, labels = dataset_generator()

    metrics = dict.fromkeys(METRICS, None)

    st = time.time()
    if k_means_kwargs["algorithm"] == 'rpkm':
        clf = RPKM(
            n_clusters=n_clusters,
            max_iter=k_means_kwargs['param'],
            n_jobs=-1
        )
        pred_labels = clf.fit_predict(data)
        distance_calculations = clf.distance_computations

    elif k_means_kwargs["algorithm"] == 'mb-kmeans':
        distance_calculations = 0.5 * n_clusters * (n_clusters + 1) * data.shape[0]
        clf = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,
            batch_size=k_means_kwargs['param'],
        )
        pred_labels = clf.fit_predict(data)
        distance_calculations += n_clusters * k_means_kwargs['param'] * clf.n_iter_

    else:
        distance_calculations = 0.5 * n_clusters * (n_clusters + 1) * data.shape[0]
        clf = KMeans(
            n_clusters=n_clusters,
            init=k_means_kwargs["algorithm"],
            n_init=1,
            n_jobs=-1
        )
        pred_labels = clf.fit_predict(data)
        distance_calculations += n_clusters * data.shape[0] * clf.n_iter_

    fit_time = time.time() - st

    for metric_name in metrics:
        metrics[metric_name] = METRICS[metric_name](data, pred_labels)

    assert list(metrics.keys()) == list(METRICS.keys()), 'metrics are not in the same order or are not equivalent'

    logging.info(f"parameters = [n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}, kwargs: {k_means_kwargs}]")
    logging.info(f"average_time={fit_time}, metrics={metrics}")
    return n_clusters, n_dims, n_samples, *k_means_kwargs.values(), *metrics.values()


if __name__ == '__main__':
    set_logger('logs/cluster_quality_artificial.log', debug=False)

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES, KWARGS, list(range(N_REPLICAS))))

    if N_THREADS == 1:
        results = []
        for args in tqdm(combinations):
            results.append(compute(*args))
    else:
        results = []
        pb = tqdm(total=len(combinations))

        def update(ans):
            results.append(ans)
            pb.update()

        with Pool(N_THREADS) as p:
            for comb in combinations:
                p.apply_async(compute, args=comb, callback=update)
            p.close()
            p.join()
        pb.close()
    results.sort()

    columns = ['n_clusters','n_dims','n_samples'] + list(KWARGS[0].keys()) + list(METRICS.keys())
    results = pd.DataFrame(results, columns=columns)
    results = results.fillna('None')
    results = results.groupby(columns[:5]).mean().reset_index()
    results.to_csv(OUT_FOLDER / 'quality_artificial_agg.csv', sep=',', index=False)

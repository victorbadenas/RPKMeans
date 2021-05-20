import logging
import os
import random
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

sys.path.append('src')

from artificial_dataset_generator import ArtificialDatasetGenerator
from clustering.kmeans import KMeans

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)
pb = None

OUT_FOLDER = Path('results/')
N_REPLICAS = 50
# N_REPLICAS = 3
N_CLUSTERS = [3, 9]
# N_CLUSTERS = [2, 3]
N_DIMS = [2, 4, 8]
# N_DIMS = [2, 3]
N_SAMPLES = [1e2, 1e3, 1e4, 1e5, 1e6]
# N_SAMPLES = [1e2, 1e3]

N_THREADS = None # int or None to use all of them

# model kwargs to be tested:
KWARGS = [
    {"init": 'kmeans++', "rpkm_max_iter": None},
    {"init": 'rpkm', "rpkm_max_iter": 1},
    {"init": 'rpkm', "rpkm_max_iter": 2},
    {"init": 'rpkm', "rpkm_max_iter": 3},
    {"init": 'rpkm', "rpkm_max_iter": 4},
    {"init": 'rpkm', "rpkm_max_iter": 5},
    {"init": 'rpkm', "rpkm_max_iter": 6},
]

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def set_logger(log_file_path, debug=False):
    log_file_path = Path(log_file_path)
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(module)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
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
    st = time.time()
    clf = KMeans(
        n_clusters=n_clusters,
        n_init=1,
        **k_means_kwargs).fit(data)

    fit_time = time.time() - st

    distance_calculations = clf.distance_computations

    y = clf.predict(data)
    conf_mat = confusion_matrix(y, labels)
    acc = np.max(conf_mat, axis=0).sum() / len(labels)

    logging.info(f"parameters = [n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}, kwargs: {k_means_kwargs}]")
    logging.info(f"distance_calculations={distance_calculations}, accuracy_average={acc}, average_time={fit_time}")
    return n_clusters, n_dims, n_samples, *k_means_kwargs.values(), acc, distance_calculations, fit_time


if __name__ == '__main__':
    set_logger('logs/artificial_dataset_tests.log', debug=False)

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES, KWARGS, list(range(N_REPLICAS))))

    if N_THREADS == 1:
        results = []
        for n_clusters, n_dims, n_samples, kwargs, computationIdx in tqdm(combinations):
            results.append(compute(n_clusters, n_dims, n_samples, kwargs, computationIdx))
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

    columns = ['n_clusters','n_dims','n_samples'] + list(KWARGS[0].keys()) + ['accuracy_average','distance_calculations_average','fit_time_average']
    results = pd.DataFrame(results, columns=columns)
    results = results.fillna('None')
    results = results.groupby(columns[:5]).mean().reset_index()
    results.to_csv(OUT_FOLDER / 'artificial.csv', sep=',', index=False)

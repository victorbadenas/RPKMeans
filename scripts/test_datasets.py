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
from tqdm import tqdm

sys.path.append('src')
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from dataset import PandasUnsupervised
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
N_SAMPLES = [4000, 12000, 40000, 120000, 400000, 1200000, 4000000]
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

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser('Script to generate metrics and results csv for a dataset')
    parser.add_argument('-i', '--dataset', type=Path, help='path to csv file', default='data/gas_sensor/ethylene_methane.csv')
    parser.add_argument('-dc', '--dropColumns', action='append', help='columns to drop from dataset', default=['Time_(seconds)','Methane_conc_(ppm)','Ethylene_conc_(ppm)'])
    parser.add_argument('-d', '--debug', action='store_true', help='toggle debug mode', default=False)
    return parser.parse_args()

def set_logger(log_file_path, debug=False):
    log_file_path = Path(log_file_path)
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    if debug:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logging.Formatter(logging_format))
        logging.getLogger().addHandler(consoleHandler)

def compute(dataset, n_clusters, n_dims, n_samples, k_means_kwargs, computationidx):
    logging.info(f'now computing: n_clusters={n_clusters}, n_dims={n_dims}, n_samples={n_samples}, k_means_kwargs={k_means_kwargs}, computationidx={computationidx}')
    accuracy_average = 0
    fit_time_average = 0
    np.random.seed(computationidx)
    random.seed(computationidx)

    data = dataset.sample(n_samples, replace=False).sample(n_dims, axis=1)
    st = time.time()
    if k_means_kwargs["algorithm"] == 'rpkm':
        clf = RPKM(
            n_clusters=n_clusters,
            max_iter=k_means_kwargs['param'],
            n_jobs=-1
        ).fit(data)
        distance_calculations = clf.distance_computations

    elif k_means_kwargs["algorithm"] == 'mb-kmeans':
        distance_calculations = n_clusters * n_clusters * data.shape[0]
        clf = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,
            batch_size=k_means_kwargs['param'],
        ).fit(data)
        distance_calculations += n_clusters * k_means_kwargs['param'] * clf.n_iter_

    else:
        distance_calculations = n_clusters * n_clusters * data.shape[0]
        clf = KMeans(
            n_clusters=n_clusters,
            init=k_means_kwargs["algorithm"],
            n_init=1,
            n_jobs=-1
        ).fit(data)
        distance_calculations += n_clusters * data.shape[0] * clf.n_iter_

    fit_time = time.time() - st
    logging.info(f"parameters = [n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}, kwargs: {k_means_kwargs}]")
    logging.info(f"distance_calculations={distance_calculations}, average_time={fit_time}")
    return n_clusters, n_dims, n_samples, *k_means_kwargs.values(), distance_calculations, fit_time


if __name__ == '__main__':
    parameters = parse_args()
    set_logger(f'logs/{parameters.dataset.stem}.log', debug=parameters.debug)

    pdataset = PandasUnsupervised(parameters.dataset).data
    if len(parameters.dropColumns) > 0:
        pdataset = pdataset.drop(parameters.dropColumns, axis=1)
    logging.info(pdataset.describe())

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES, KWARGS, list(range(N_REPLICAS))))

    if N_THREADS == 1:
        results = []
        for args in tqdm(combinations):
            results.append(compute(pdataset, *args))
    else:
        from functools import partial
        results = []
        pb = tqdm(total=len(combinations))

        def update(ans):
            results.append(ans)
            pb.update()

        with Pool(N_THREADS) as p:
            for comb in combinations:
                p.apply_async(partial(compute, pdataset), args=comb, callback=update)
            p.close()
            p.join()
        pb.close()
    results.sort()

    columns = ['n_clusters','n_dims','n_samples'] + list(KWARGS[0].keys()) + ['distance_calculations_average','fit_time_average']
    results = pd.DataFrame(results, columns=columns)
    results = results.fillna('None')
    results.to_csv(OUT_FOLDER / f'{parameters.dataset.stem}.csv', sep=',', index=False)
    results = results.groupby(columns[:5]).mean().reset_index()
    results.to_csv(OUT_FOLDER / f'{parameters.dataset.stem}_agg.csv', sep=',', index=False)

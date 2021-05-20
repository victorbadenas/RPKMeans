import logging
import os
import random
import argparse
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

sys.path.append('src')

from dataset import PandasUnsupervised
from clustering.kmeans import KMeans

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)

OUT_FOLDER = Path('results/')
N_REPLICAS = 10
N_CLUSTERS = [3, 9]
N_DIMS = [2, 4, 8]
N_SAMPLES = [4000, 12000, 40000, 120000, 400000, 1200000, 4000000]
N_THREADS = 1 # int or None to use all of them

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser('Script to generate metrics and results csv for a dataset')
    parser.add_argument('-i', '--dataset', type=Path, help='path to csv file', default='data/gas_sensor/ethylene_methane.csv')
    parser.add_argument('-d', '--dropColumns', action='append', help='columns to drop from dataset', default=['Time_(seconds)','Methane_conc_(ppm)','Ethylene_conc_(ppm)'])
    return parser.parse_args()

def set_logger(log_file_path, debug=False):
    log_file_path = Path(log_file_path)
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

def compute(dataset, n_clusters, n_dims, n_samples):
    distance_calculations_average = 0
    accuracy_average = 0
    fit_time_average = 0

    for _ in range(N_REPLICAS):
        data = dataset.sample(n_samples, replace=False).sample(n_dims, axis=1)
        st = time.time()
        clf = KMeans(
            n_clusters=n_clusters,
            init='rpkm',
            n_init=1).fit(data)

        fit_time_average += time.time() - st
        distance_calculations_average += clf.distance_computations

    distance_calculations_average /= N_REPLICAS
    accuracy_average /= N_REPLICAS
    fit_time_average /= N_REPLICAS

    logging.info(f"parameters = [n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}]")
    logging.info(f"distance_calculations_average={distance_calculations_average}, accuracy_average={accuracy_average}, average_time={fit_time_average}")
    return n_clusters, n_dims, n_samples, accuracy_average, distance_calculations_average, fit_time_average


if __name__ == '__main__':
    args = parse_args()
    set_logger(f'logs/{args.dataset.stem}.log', debug=False)

    pdataset = PandasUnsupervised(args.dataset).data
    if len(args.dropColumns) > 0:
        pdataset = pdataset.drop(args.dropColumns, axis=1)
    pdataset = 2*(pdataset - 0.5)
    logging.info(pdataset.describe())

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES))

    if N_THREADS == 1:
        results = []
        for n_clusters, n_dims, n_samples in tqdm(combinations):
            results.append(compute(pdataset, n_clusters, n_dims, n_samples))
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

    with open(OUT_FOLDER / f'{args.dataset.stem}.csv', 'w') as f:
        f.write('n_clusters,n_dims,n_samples,accuracy_average,distance_calculations_average,fit_time_average\n')
        f.write('\n'.join(map(lambda x: ','.join(map(str, x)), results)))

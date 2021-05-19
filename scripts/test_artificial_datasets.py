import logging
import os
import random
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

sys.path.append('src')

from artificial_dataset_generator import ArtificialDatasetGenerator
from clustering.kmeans import KMeans

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)

OUT_FOLDER = Path('results/')
N_REPLICAS = 50
N_CLUSTERS = [3, 9]
N_DIMS = [2, 4, 8]
N_SAMPLES = [1e3, 1e4, 1e5, 1e6]
# N_SAMPLES = [1e3, 1e4]
N_THREADS = None # int or None to use all of them

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

def set_logger(log_file_path, debug=False):
    log_file_path = Path(log_file_path)
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

def compute(n_clusters, n_dims, n_samples):
    dataset_generator = ArtificialDatasetGenerator(
        n_centers=n_clusters,
        n_features=n_dims,
        n_samples=int(n_samples),
        normalize=True,
        n_replicas=N_REPLICAS)

    distance_calculations_average = 0
    accuracy_average = 0

    for i, (data, labels) in enumerate(dataset_generator):
        clf = KMeans(
            n_clusters=n_clusters,
            init='rpkm',
            n_init=1).fit(data)

        distance_calculations_average += clf.distance_computations

        y = clf.predict(data)
        conf_mat = confusion_matrix(y, labels)
        acc = np.max(conf_mat, axis=0).sum() / len(labels)
        accuracy_average += acc

        # logging.info(f'replica {i}: accuracy={acc} distance_computations={clf.distance_computations}')

    distance_calculations_average /= (i+1)
    accuracy_average /= (i+1)

    logging.info(f"parameters = [n_clusters: {n_clusters}, n_dims: {n_dims}, n_samples: {n_samples}]")
    logging.info(f"distance_calculations_average={distance_calculations_average}, accuracy_average={accuracy_average}")
    return n_clusters, n_dims, n_samples, accuracy_average, distance_calculations_average


if __name__ == '__main__':
    set_logger('logs/artificial_dataset_tests.log', debug=False)

    combinations = list(product(N_CLUSTERS, N_DIMS, N_SAMPLES))

    if N_THREADS == 1:
        results = []
        for n_clusters, n_dims, n_samples in tqdm(combinations):
            results.append(compute(n_clusters, n_dims, n_samples))
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

    with open(OUT_FOLDER / 'artificial.csv', 'w') as f:
        f.write('n_clusters,n_dims,n_samples,accuracy_average,distance_calculations_average\n')
        f.write('\n'.join(map(lambda x: ','.join(map(str, x)), results)))

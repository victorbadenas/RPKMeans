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
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm

sys.path.append('src')
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from dataset import PandasUnsupervised, PandasDataset
from cluster import RPKM
from sklearn.cluster import KMeans, MiniBatchKMeans

SEED = 1995
random.seed(SEED)
np.random.seed(SEED)
pb = None

OUT_FOLDER = Path('results/')
OUT_FOLDER.mkdir(parents=True, exist_ok=True)

N_REPLICAS = 2

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

def compute(X, y, n_clusters, computationIdx):
    np.random.seed(computationIdx)
    random.seed(computationIdx)

    clf = RPKM(
        n_clusters=n_clusters,
        max_iter=6,
        n_jobs=-1
    )

    ref_clf = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=1,
        n_jobs=-1
    )
    pred_labels = clf.fit_predict(X)
    ref_labels = ref_clf.fit_predict(X)

    distance_calculations = clf.distance_computations
    ref_distance_calculations = 0.5 * n_clusters * (n_clusters + 1) * X.shape[0] + n_clusters * X.shape[0] * clf.n_iter_

    m = clf.n_iter_
    instance_ratio = clf.instance_ratio

    emast = np.sum(cdist(X, ref_clf.cluster_centers_).min(axis=1)**2)
    em = np.sum(cdist(X, clf.centroids).min(axis=1)**2)
    std_error = (emast - em) / emast

    ami = adjusted_mutual_info_score(ref_labels, pred_labels)

    sil = 0.5*(silhouette_score(X, pred_labels) + 1)
    ref_sil = 0.5*(silhouette_score(X, ref_labels) + 1)
    sil = abs(sil - ref_sil) / (sil + ref_sil)

    chs = calinski_harabasz_score(X, pred_labels)
    ref_chs = calinski_harabasz_score(X, ref_labels)
    chs = abs(chs - ref_chs) / (chs + ref_chs)

    dbs = davies_bouldin_score(X, pred_labels)
    ref_dbs = davies_bouldin_score(X, ref_labels)
    dbs = abs(dbs - ref_dbs) / (dbs + ref_dbs)


    return *X.shape, m, distance_calculations, ref_distance_calculations, instance_ratio, std_error, ami, sil, chs, dbs


if __name__ == '__main__':
    # set_logger('log/real_world_datasets.log', debug=True)

    results = list()
    files = list(Path('data/real-world/').glob('*.csv'))
    for csv_path in tqdm(files):
        dataset = PandasDataset(csv_path)
        x, y = dataset.x, dataset.y
        n_clusters = len(set(y))
        for i in range(N_REPLICAS):
            line = compute(x, y, n_clusters, i)
            line = [dataset.name] + list(line)
            results.append(line)
    results.sort()
    results = pd.DataFrame(results, columns=['dataset', 'instances', 'dimensions', 'iterations', 'RPKM_distances', 'k-means++_distances', 'instance_ratio', 'stderror', 'ami', 'sil_error', 'chs_error', 'dbs_error'])
    results = results.fillna('None')
    results = results.groupby(['dataset','instances','dimensions']).mean().reset_index()
    results.to_csv(OUT_FOLDER / 'real_datasets.csv', sep=',', index=False)

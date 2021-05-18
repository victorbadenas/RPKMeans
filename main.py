import sys
import argparse
import logging
import numpy as np
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix

random.seed(1995)
np.random.seed(1995)

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from clustering.kmeans import KMeans
from artificial_dataset_generator import ArtificialDatasetGenerator


def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-l', "--logger", type=Path, default="log.log")
    parser.add_argument('-i', '--dataset', type=Path, default="artificial")
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    return parser.parse_args()


class Main:
    def __init__(self, parameters):
        logging.info(parameters)
        self.parameters = parameters

    def __call__(self, *args, **kwargs):
        if str(self.parameters.dataset) == 'artificial':
            adg = ArtificialDatasetGenerator(n_centers=4, n_features=2, n_samples=10000, normalize=True, cluster_std=.1)
            data, labels = adg()
        else:
            data, labels = self.load_supervised(self.parameters.dataset, skip_headers=True)
        kmeans = KMeans(n_clusters=4, init='rpkm', n_init=1, verbose=self.parameters.debug)
        pred_labels = kmeans.fit_predict(data)
        conf_mat = confusion_matrix(pred_labels, labels)
        logging.info(f'confusion_matrix:\n{conf_mat}')
        logging.info(f'precision: {np.max(conf_mat, axis=0).sum() / np.sum(conf_mat)}')

    @staticmethod
    def load_supervised(path, skip_headers=False):
        data = np.genfromtxt(path, dtype=str, skip_header=1 if skip_headers else 0, delimiter=',')
        return data[:,:-1].astype(np.float16), data[:,-1]


if __name__ == "__main__":
    parameters = parseArgumentsFromCommandLine()
    set_logger(parameters.logger)
    Main(parameters)()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import argparse
from pathlib import Path

colors = list(mcolors.TABLEAU_COLORS.values())


def parse_args():
    parser = argparse.ArgumentParser('Script to generate plots from a results csv')
    parser.add_argument('-i', '--csvPath', type=Path, help='path to csv file', default='results/artificial.csv')
    parser.add_argument('-t', '--targetvar', type=str, help='csv variable to plot', default='distance_calculations_average')
    parser.add_argument('-n', '--plotName', type=Path, help='path to store the plot', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = pd.read_csv(args.csvPath)

    x_name = 'n_samples'
    var_name = args.targetvar
    m_name = 'n_clusters'
    n_name = 'n_dims'
    model_kwargs = ['algorithm', 'param']

    M = data[m_name].unique()
    N = data[n_name].unique()

    f, axes = plt.subplots(len(N), len(M), figsize=(15, 15), sharex=True)

    for i, n in enumerate(N):
        for j, m in enumerate(M):
            ax_data = data[data[n_name] == n][data[m_name] == m].sort_values(by=x_name)

            # todo add different models
            models = sorted(set(zip(*[data[kw] for kw in model_kwargs])))
            for color_idx, model in enumerate(models):
                model = dict(zip(model_kwargs, model))
                ax_data_ = ax_data.copy()
                for k, v in model.items():
                    ax_data_ = ax_data_[ax_data_[k] == v]
                x = ax_data_[x_name]
                y = ax_data_[var_name]
                axes[i][j].loglog(x, y, label=":".join(map(str, model.values())), marker='x', color=colors[color_idx%len(colors)])
            axes[i][j].grid('on')
            # axes[i][j].legend()

            if i == len(N) - 1:
                axes[i][j].set_xlabel(x_name)
            if j == 0:
                axes[i][j].set_ylabel(var_name)
            axes[i][j].set_title(f'{n_name}:{n}, {m_name}:{m}')

    handles, labels = axes[i][j].get_legend_handles_labels()
    f.legend(handles, labels)
    f.suptitle(f'{var_name} vs {x_name}', fontsize=16)
    # plt.tight_layout()
    plt.savefig(args.plotName, dpi=400)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import argparse
from pathlib import Path

colors = list(mcolors.TABLEAU_COLORS.values())


def parse_args():
    parser = argparse.ArgumentParser('Script to generate plots from a results csv')
    parser.add_argument('-i', '--csvPath', type=Path, help='path to csv file', default='results/instance_ratio_agg.csv')
    parser.add_argument('-t', '--targetvar', type=str, help='csv variable to plot', default='instance_ratio')
    parser.add_argument('-n', '--plotName', type=Path, help='path to store the plot', default='images/debug.png')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = pd.read_csv(args.csvPath)

    x_name = 'param'
    var_name = args.targetvar
    m_name = 'n_clusters'
    n_name = 'n_dims'

    M = data[m_name].unique()
    N = data[n_name].unique()

    f, axes = plt.subplots(len(N), len(M), figsize=(15, 15), sharex=True)

    for i, n in enumerate(N):
        for j, m in enumerate(M):
            ax_data = data[data[n_name] == n][data[m_name] == m].sort_values(by=x_name)

            # todo add different models
            for color_idx, n_samples in enumerate(ax_data.n_samples.unique()):
                ax_data_ = ax_data[ax_data.n_samples == n_samples]
                x = ax_data_[x_name]
                y = ax_data_[var_name]
                axes[i][j].plot(x, y, label=str(n_samples), marker='x', color=colors[color_idx%len(colors)])
            axes[i][j].grid('on')

            if i == len(N) - 1:
                axes[i][j].set_xlabel('m')
            if j == 0:
                axes[i][j].set_ylabel('|P|/n')
            axes[i][j].set_title(f'{n_name}:{n}, {m_name}:{m}')

    handles, labels = axes[i][j].get_legend_handles_labels()
    f.legend(handles, labels)
    f.suptitle(f'{var_name} vs {x_name}', fontsize=16)
    # plt.tight_layout()
    plt.savefig(args.plotName, dpi=400)
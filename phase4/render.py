'''
'''

import torch
from torch import logical_and, logical_not, logical_or
from torch import nn
import numpy as np
import torchmetrics as tm
from matplotlib import pyplot
import seaborn as sn
from .utilities import Percentile
# from sklearn.metrics import confusion_matrix as ccmat

from matplotlib import rcParams

def plot_metrics(metrics: dict, fig_name, logger, RT):

    cm = metrics.pop('ConfusionMatrix')
    cm = cm.cpu()

    n_cols = 6  # for subplots
    n_rows = int(np.ceil(len(cm) / n_cols))
    fig, ax_array = pyplot.subplots(                                 # ax_array[i,j]
        n_rows, n_cols, figsize=(16, 9),
        sharex=True, sharey=True, constrained_layout=True,
        gridspec_kw={'wspace': .2, 'hspace': .2},
    )

    ax_index = 0
    # used to fill bottom subplots first (because they are used by sharex=True parameter)
    for i in range(n_rows-1, -1, -1):
        for j in range(n_cols):

            if ax_index < len(cm):
                sn.heatmap(cm[ax_index],
                           ax=ax_array[i, j], cmap='Blues', annot=True, fmt=".2f", center=None, vmin=0,
                           xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"],
                           )
                # ax_list[i,j].xaxis.tick_top()
                ax_array[i, j].set_title(f'DOA={(ax_index) * 5}')
                if j == 0:
                    # only for first column (because it is used by sharey=True parameter)
                    ax_array[i, j].set_ylabel('Prediction')
                if i == n_rows-1:
                    # only for last row (because it is used by sharex=True parameter)
                    ax_array[i, j].set_xlabel('Target')

            else:
                # hide unused subplots
                ax_array[i, j].set_visible(False)
            ax_index += 1

    # fig_title = f"Confusion Matrix_Multiclass_MultiLabel_RT={RT} - Prec: {np.mean(metrics['Precision'][1:]):.2f} - Recall: {np.mean(metrics['Recall'][1:]):.2f}"
    fig_title = f"Confusion Matrix Multiclass MultiLabel RT={RT} - Prec: {metrics['Precision']:.2f} - Recall: {metrics['Recall']:.2f}"
    # for key, metric in metrics.items():
    #     fig_title += f"\n{key}:  [{',  '.join([f'{m:.2f}' for m in metric[1:]  ])}]"

    fig.suptitle(fig_title)

    fig.savefig(f"{logger}/{fig_name}", dpi=100, bbox_inches='tight')
    # pyplot.show()
    pyplot.close(fig)


def plot_prediction_v2(pred, labels, t_total):
    fig, (ax1, ax2) = pyplot.subplots(2, 1, sharex=True)

    ax1.matshow(labels, aspect='auto', cmap='Greys', origin='lower',
                vmin=0, vmax=1, extent=(0, t_total, 0, 180))
    ax1.set(title="True Labels", ylabel='Angle')
    ax1.xaxis.set_ticks_position('bottom')

    ax2.matshow(pred, aspect='auto', cmap='Greys',
                origin='lower', vmin=0, vmax=1, extent=(0, t_total, 0, 180))
    ax2.set(title="Raw Predictions", ylabel='Angle')
    ax2.xaxis.set_ticks_position('bottom')

    # pyplot.show()


def plot_prediction(bin_targets: torch.Tensor, predictions: torch.Tensor, bin_pred: torch.Tensor, 
                metrics: dict, win_size, log_dir: str, model_name):

    # with open(f"{self.logger.log_dir}/metrics.txt", 'a') as f:
    #     f.write(f"Test Batch: {batch_idx}\n")
    #     [f.write(f"{k}: {v}\n") for k, v in metrics.items()]
    #     f.write(f"\n")

    # Post-processing
    percentile = Percentile(.75)
    smooth_bin_pred = percentile(bin_pred)

    # Plot Predictions
    # fig, (ax1, ax2, ax3) = pyplot.subplots(
    pad = win_size / 2
    file_len_sec = 50
    N = int((file_len_sec - win_size + 2*pad) / (.5*win_size) )

    for i in np.arange(0, len(predictions), N):
        if len(predictions[i:i+N]) < N:
            continue

        ts = 0 if i == 0 else i/N*file_len_sec
        te = (i+N)/N*file_len_sec
        fig, (ax1, ax2) = pyplot.subplots(
            2, 1, sharex=True, figsize=(12, 10), squeeze=True)
        ax1.matshow(bin_targets[i:i+N].T.cpu().numpy(),
                    aspect='auto', origin='lower',
                    vmin=0, vmax=1, cmap='Greys', extent=[ts, te, 0, 180])
        ax1.set(title="Targets (one-hot-encoded)", ylabel='DOA [degrees]')
        ax1.xaxis.set_ticks_position('bottom')

        ax2.matshow(predictions[i:i+N].T.cpu().numpy(),
                    aspect='auto', origin='lower',
                    vmin=0, vmax=1, cmap='Greys', extent=[ts, te, 0, 180])
        ax2.set(title="Predictions (Probabilities)", ylabel='DOA [degrees]')
        ax2.xaxis.set_ticks_position('bottom')


        fig.suptitle(f"Prediction Plot of {model_name}")
        # fig.suptitle(f"Prediction Plot of {name} \n Metrics = {metrics}")

        # if batch_idx == 0:
        #     pyplot.show()
        fig.savefig(
            f"{log_dir}/predictions_{i}.png", dpi=600)

        pyplot.close(fig)

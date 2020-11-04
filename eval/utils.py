import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from utils.utils import getLabel


def evaluate(model, loader, 
                mode='default', verbose=1):

    """
        Evaluate model on dataset
    """

    n = len(loader)
    y_tot, y_pred_tot = [], []

    model.eval()
    for batch_idx, (x, y) in enumerate(loader):

        y = getLabel(y, mode=mode)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        y_tot.append(y.detach().cpu().numpy())
        y_pred_tot.append(y_pred.detach().cpu().numpy())

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Evaluate - [{}>{}] ({}/{})".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n
            )
            print(progress, end='\r')

    print()

    y_tot, y_pred_tot = np.concatenate(y_tot), np.concatenate(y_pred_tot)

    n_ex = y_tot.shape[0]
    x_range = np.arange(0, n_ex, 1.0)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_tot[:, 0], y_tot[:, 1], label='GT')
    plt.scatter(y_pred_tot[:, 0], y_pred_tot[:, 1], label='Pred')
    plt.close()

    logg = {
        'fig': wandb.Image(fig),
    }

    return logg
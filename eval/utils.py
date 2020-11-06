import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from utils.vis import plotArrMulti
from utils.utils import getLabel
from eval.metrics import ErrAcc

def evaluate(model, loader, 
                mode='default', verbose=1,
                rel_err_acc_meters=[1, 5, 10],
                abs_err_acc_meters=[1, 5, 10]):

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

    r1s = [y_tot[:, 0], y_pred_tot[:, 0]]
    r2s = [y_tot[:, 1], y_pred_tot[:, 1]]
    # label_r1s = ['r1_gt', 'r1_pred']
    # label_r2s = ['r2_gt', 'r2_pred']
    # fig_r1 = plotArrMulti(r1s, labels=label_r1s, debug=False, scatter=True)
    # fig_r2 = plotArrMulti(r2s, labels=label_r2s, debug=False, scatter=True)

    # logg = {
    #     'fig': wandb.Image(fig),
    #     'fig_r1': wandb.Image(fig_r1),
    #     'fig_r2': wandb.Image(fig_r2)
    # }

    loggs = []
    n = len(r1s[0])
    for i in range(n):
        logg = {
            'r1': r1s[0][i],
            'r1_pred': r1s[1][i],
            'r2': r2s[0][i],
            'r2_pred': r2s[1][i]
        }
        loggs.append(logg)

    err_acc = {}
    rel_err_acc_r1 = ErrAcc(
        r1s[0],
        r1s[1],
        err=rel_err_acc_meters,
        err_mode='rel',
        keyPrefix='r1'
    )
    rel_err_acc_r2 = ErrAcc(
        r2s[0],
        r2s[1],
        err=rel_err_acc_meters,
        err_mode='rel',
        keyPrefix='r2'
    )
    abs_err_acc_r1 = ErrAcc(
        r1s[0],
        r1s[1],
        err=abs_err_acc_meters,
        err_mode='abs',
        keyPrefix='r1',
        data_factor=100
    )
    abs_err_acc_r2 = ErrAcc(
        r2s[0],
        r2s[1],
        err=abs_err_acc_meters,
        err_mode='abs',
        keyPrefix='r2',
        data_factor=100
    )
    
    err_acc.update(rel_err_acc_r1)
    err_acc.update(rel_err_acc_r2)
    err_acc.update(abs_err_acc_r1)
    err_acc.update(abs_err_acc_r2)

    loggs.append(err_acc)

    return loggs
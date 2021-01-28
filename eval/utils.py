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

    r1_idx, r2_idx = None, None
    e1r_idx, e1i_idx, e3r_idx, e3i_idx = [None]*4

    if mode=='r':
        r1_idx, r2_idx = 0, 1


    rel_err_acc_r1, rel_err_acc_r2, abs_err_acc_r1, abs_err_acc_r2 = [None]*4
    rel_err_acc_r1_calculator = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r1')
    rel_err_acc_r2_calculator = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r2')
    abs_err_acc_r1_calculator = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r1', data_factor=100)
    abs_err_acc_r2_calculator = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r2', data_factor=100)

    model.eval()
    for batch_idx, (x, y) in enumerate(loader):

        y = getLabel(y, mode=mode)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        y = y.detach().cpu()
        y_pred = y_pred.detach().cpu()
        rel_err_acc_r1_calculator.feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
        rel_err_acc_r2_calculator.feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())
        abs_err_acc_r1_calculator.feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
        abs_err_acc_r2_calculator.feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())

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

    r1s = [y_tot[:, 0], y_pred_tot[:, 0]]
    r2s = [y_tot[:, 1], y_pred_tot[:, 1]]

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

    rel_err_acc_r1 = rel_err_acc_r1_calculator.getAcc()
    rel_err_acc_r2 = rel_err_acc_r2_calculator.getAcc()
    abs_err_acc_r1 = abs_err_acc_r1_calculator.getAcc()
    abs_err_acc_r2 = abs_err_acc_r2_calculator.getAcc()
    
    err_acc.update(rel_err_acc_r1)
    err_acc.update(rel_err_acc_r2)
    err_acc.update(abs_err_acc_r1)
    err_acc.update(abs_err_acc_r2)

    loggs.append(err_acc)

    return loggs
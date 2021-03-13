from collections import defaultdict
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from utils.vis import plotArrMulti
from utils.utils import getLabel, isMode, transform_domain
from eval.metrics import ErrAcc

def evaluate(model, loader, 
                mode='default', verbose=1,
                rel_err_acc_meters=[1, 5, 10],
                abs_err_acc_meters=[1, 5, 10],
                e1_classes=None,
                domain=0):

    """
        Evaluate model on dataset
    """

    n = len(loader)
    y_tot, y_pred_tot = [], []
    if isMode(mode, 'e1'):
        y_tot_e1, y_pred_tot_e1 = defaultdict(list), defaultdict(list)

    # r1_idx, r2_idx = None, None
    # e1r_idx, e1i_idx, e3r_idx, e3i_idx = [None]*4

    r1_idx, r2_idx = 0, 1

    rel_err_acc_r1, rel_err_acc_r2, abs_err_acc_r1, abs_err_acc_r2 = [None]*4
    if isMode(mode, 'e1'):
        DEFAULT_KEY = 'default'
        rel_err_acc_r1_calculator = {DEFAULT_KEY: ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r1')}
        rel_err_acc_r2_calculator = {DEFAULT_KEY: ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r2')}
        abs_err_acc_r1_calculator = {DEFAULT_KEY: ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r1')}
        abs_err_acc_r2_calculator = {DEFAULT_KEY: ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r2')}
        for class_ in e1_classes:
            rel_err_acc_r1_calculator[f'{class_}'] = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix=f'{class_}_r1')
            rel_err_acc_r2_calculator[f'{class_}'] = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix=f'{class_}_r2')
            abs_err_acc_r1_calculator[f'{class_}'] = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix=f'{class_}_r1')
            abs_err_acc_r2_calculator[f'{class_}'] = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix=f'{class_}_r2')
    else:
        rel_err_acc_r1_calculator = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r1')
        rel_err_acc_r2_calculator = ErrAcc(mode='rel', err=rel_err_acc_meters, keyPrefix='r2')
        abs_err_acc_r1_calculator = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r1', data_factor=100)
        abs_err_acc_r2_calculator = ErrAcc(mode='abs', err=abs_err_acc_meters, keyPrefix='r2', data_factor=100)

    model.eval()

    for batch_idx, (x, y) in enumerate(loader):

        # y = getLabel(y, mode=mode)

        if isMode(mode, 'e1'):
            x, x_e1 = x

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        if isMode(mode, 'e1'):
            y_pred = model(x, x_e1)
        else:
            y_pred = model(x)

        y = y.detach().cpu()
        y_pred = y_pred.detach().cpu()
        y_pred = transform_domain(y_pred, domain=domain, reverse_=True)

        # Calculate metric on the fly
        if isMode(mode, 'e1'):
            rel_err_acc_r1_calculator['default'].feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            rel_err_acc_r2_calculator['default'].feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())
            abs_err_acc_r1_calculator['default'].feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            abs_err_acc_r2_calculator['default'].feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())

            rel_err_acc_r1_calculator[f'{x_e1}'].feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            rel_err_acc_r2_calculator[f'{x_e1}'].feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())
            abs_err_acc_r1_calculator[f'{x_e1}'].feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            abs_err_acc_r2_calculator[f'{x_e1}'].feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())
        else:
            rel_err_acc_r1_calculator.feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            rel_err_acc_r2_calculator.feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())
            abs_err_acc_r1_calculator.feedData(y[:, r1_idx].numpy(), y_pred[:, r1_idx].numpy())
            abs_err_acc_r2_calculator.feedData(y[:, r2_idx].numpy(), y_pred[:, r2_idx].numpy())

        # Keep track of predictions vs labels
        y_tot.append(y.numpy())
        y_pred_tot.append(y_pred.numpy())
        if isMode(mode, 'e1'):
            y_tot_e1[x_e1].append(y.numpy())
            y_pred_tot_e1[x_e1].append(y_pred.numpy())

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Evaluate - [{}>{}] ({}/{})".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n
            )
            print(progress, end='\r')

    print()

    # Process total data [labels & predictions]
    y_tot, y_pred_tot = np.concatenate(y_tot), np.concatenate(y_pred_tot)
    if isMode(mode, 'e1'):
        for class_ in e1_classes:
            y_tot_e1[class_] = np.concatenate(y_tot_e1[class_])
            y_pred_tot_e1[class_] = np.concatenate(y_pred_tot_e1[class_])

    # Format into r1 & r2
    r1s = [y_tot[:, 0], y_pred_tot[:, 0]]
    r2s = [y_tot[:, 1], y_pred_tot[:, 1]]
    if isMode(mode, 'e1'):
        r1s_e1, r2s_e1 = {}, {}
        for class_ in e1_classes:
            r1s_e1[class_] = [y_tot_e1[class_][:, 0], y_pred_tot_e1[class_][:, 0]]
            r2s_e1[class_] = [y_tot_e1[class_][:, 1], y_pred_tot_e1[class_][:, 1]]

    # Prepare loggs to return
    loggs = []

    # Append predictions vs labels
    n = len(r1s[0])
    for i in range(n):
        logg = {
            'r1': r1s[0][i],
            'r1_pred': r1s[1][i],
            'r2': r2s[0][i],
            'r2_pred': r2s[1][i]
        }
        loggs.append(logg)
    if isMode(mode, 'e1'):
        for class_ in e1_classes:
            n = len(r1s_e1[class_][0])
            for i in range(n):
                logg = {
                    f'r1_{class_}': r1s_e1[class_][0][i],
                    f'r1_pred_{class_}': r1s_e1[class_][1][i],
                    f'r2_{class_}': r2s_e1[class_][0][i],
                    f'r2_pred_{class_}': r2s_e1[class_][1][i],
                }
                loggs.append(logg)

    # Append metric values into loggs
    err_acc = {}

    if isMode(mode, 'e1'):
        for class_ in e1_classes+[DEFAULT_KEY]:
            rel_err_acc_r1 = rel_err_acc_r1_calculator[f'{class_}'].getAcc()
            rel_err_acc_r2 = rel_err_acc_r2_calculator[f'{class_}'].getAcc()
            abs_err_acc_r1 = abs_err_acc_r1_calculator[f'{class_}'].getAcc()
            abs_err_acc_r2 = abs_err_acc_r2_calculator[f'{class_}'].getAcc()

            err_acc.update(rel_err_acc_r1)
            err_acc.update(rel_err_acc_r2)
            err_acc.update(abs_err_acc_r1)
            err_acc.update(abs_err_acc_r2)
    else:
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

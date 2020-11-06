import numpy as np
import torch


def ErrAcc(y, y_pred, 
        err=[1, 5, 10], err_mode="abs",
        keyPrefix='r', data_factor=100):
    """
        Calculates accuracy with maximum allowable error
        Args:
            y : Ground truth
            y_pred : Predictions
            err : Maximum allowable error
            err_mode : Error Mode, rel (relative percentage) or abs
            keyPrefix : prefix of output dictionary key
    """

    if err_mode=='abs':
        d = abs(y - y_pred) * data_factor
    else:
        d = abs(100 * (y - y_pred) / y)

    res = {}
    for err_ in err:
        n, n_correct = d.shape[0], (d<err_).sum()
        if err_mode=='abs':
            res['ErrAcc_abs_{}_{}'.format(keyPrefix, err_)] = n_correct / n
        else:
            res['ErrAcc_rel_{}_{}'.format(keyPrefix, err_)] = n_correct / n

    return res
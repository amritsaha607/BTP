import numpy as np
import torch


def ErrAcc(y, y_pred, err=[1, 5, 10], keyPrefix='r'):
    """
        Calculates accuracy with maximum allowable error
        Args:
            y : Ground truth
            y_pred : Predictions
            err : Maximum allowable percentage error
    """

    d = abs(100 * (y - y_pred) / y)
    res = {}
    for err_ in err:
        n, n_correct = d.shape[0], (d<err_).sum()
        res['ErrAcc_{}_{}'.format(keyPrefix, err_)] = n_correct / n
    return res
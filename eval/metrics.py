import numpy as np
import torch


class ErrAcc:

    """
        Calculates accuracy with maximum allowable error
    """

    def __init__(self, mode, 
        err=[1, 5, 10],
        keyPrefix='r', data_factor=100):
        """
            Args:
                mode : 'abs' or 'rel' for absolute and relative error
                        Error Mode, rel (relative percentage) or abs
                err : Maximum allowable error
                keyPrefix : prefix of output dictionary key
        """
        n_err = len(err)
        self.n_correct = [0]*n_err
        self.n = [0]*n_err
        self.acc = {}
        self.mode = mode
        self.err = err
        self.keyPrefix = keyPrefix
        self.data_factor = data_factor

    def feedData(self, y, y_pred):
        """
            Calculates accuracy with maximum allowable error
            Args:
                y : Ground truth
                y_pred : Predictions
        """
        if self.mode=='abs':
            d = abs(y - y_pred) * self.data_factor
        else:
            d = abs(100 * (y - y_pred) / y)

        for idx, err_ in enumerate(self.err):
            n, n_correct = d.shape[0], (d<err_).sum()
            self.n_correct[idx] += n_correct
            self.n[idx] += n
            self.acc['ErrAcc_{}_{}_{}'.format(self.mode, self.keyPrefix, err_)] = self.n_correct[idx] / self.n[idx]

        return self.acc

    def getAcc(self):
        return self.acc

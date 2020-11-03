import torch
import torch.nn as nn


class SeperateLoss(nn.Module):

    """
        Calculates seperated loss between prediction and ground truth values
        w.r.t. different parameters (viz. r1, r2, e1, e2, e3)
    """

    def __init__(self):
        super(SeperateLoss, self).__init__()
        

    def forward(self, y_pred, y, 
                mode='mse', run='train', weights=None):
        """
            Forward pass through loss
            Args:
                y_pred, y : 2n+6 X 1 tensors
                mode : 
                    'MSE' : returns MSELoss
                    'split': returns splitted loss for all params as a dict
                                (r1, r2, e1, e2, e3)
                    'split_re': returns splitted loss for r & e as a dict
                                (r, e)
                    'weighted': returns weighted loss with provided loss
        """

        batch_size, n = y_pred.shape
        d = (y_pred - y)**2

        if mode=='mse':
            loss = d.sum() / (batch_size*n)

        elif mode.startswith('split'):

            if 'weight' in mode:
                d *= weights

            if 'split_re' in mode:
                loss = {
                    '{}_loss_r'.format(run): d[:, :2].sum() / (batch_size*2),
                    '{}_loss_e'.format(run): d[:, 2:].sum() / (batch_size*(n-2))
                }
            else:
                n = (n-6) // 2
                loss = {
                    '{}_loss_r1'.format(run): d[:, 0].sum() / (batch_size),
                    '{}_loss_r2'.format(run): d[:, 1].sum() / (batch_size),
                    '{}_loss_e1'.format(run): d[:, 2:4].sum() / (batch_size*2),
                    '{}_loss_e3'.format(run): d[:, 4:6].sum() / (batch_size*2),
                    '{}_loss_e2_r'.format(run): d[:, 6:6+n].sum() / (batch_size*n),
                    '{}_loss_e2_i'.format(run): d[:, 6+n:].sum() / (batch_size*n)
                }

        elif mode=='weighted':
            d = d * weights
            loss = d.sum() / (batch_size*n)

        return loss

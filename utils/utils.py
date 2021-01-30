import numpy as np
import torch

def getLabel(y, mode='default'):
    """
        Extracts needed labels from given data of all labels
        Args:
            y   : all labels
            mode:
                default: all output
                r: radius
                eps_sm: only e1 & e3 (both real and imaginary parts)
                eps: epsilon data
    """
    if mode=='default':
        return y
    elif mode=='r':
        y = y[:, :2]
        return y
    elif mode=="eps_sm":
        y = y[:, 2:6]
        return y
    elif mode=='eps':
        y = y[:, 2:]
        return y
    else:
        raise ValueError("Unknown mode {} found".format(mode))


def getPredictions(y):

    '''
        Returns prediction dict from raw model output
        Args:
            y : model output
                shape : (batch_size X (2n+6))
    '''
    
    batch_size, tot_dim = y.shape[0], y.shape[1]
    n = (tot_dim - 6) // 2
    r1 = y[:, 0:1]
    r2 = y[:, 1:2]
    e1 = y[:, 2:4]
    e3 = y[:, 4:6]
    e2_r = y[:, 6:6+n]
    e2_i = y[:, 6+n:]
    res = {
        'r1': r1,
        'r2': r2,
        'e1': e1,
        'e3': e3,
        'e2_r': e2_r,
        'e2_i': e2_i
    }
    return res

def getLossWeights(weights_dict, n):
    """
        Get loss weights from loss_weights config dict
        Args:
            weights_dict : Contains weights of different parameters
            n : number of samples
    """

    w = torch.ones(2*n+6,)
    w[0] *= weights_dict['r1']
    w[1] *= weights_dict['r2']
    w[2] *= weights_dict['e1_r']
    w[3] *= weights_dict['e1_i']
    w[4] *= weights_dict['e3_r']
    w[5] *= weights_dict['e3_i']
    w[6:6+n] *= weights_dict['e2_r']
    w[6+n:] *= weights_dict['e2_i']
    return w


def oneHot(index, n):
    """
        Creates one hot vector of n classes with "index" class
    """
    x = np.zeros(n)
    x[index] = 1
    return x

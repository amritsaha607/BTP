from utils.utils import isMode
import numpy as np
import torch

def collateR(data):
    """
        Collates data for pytorch dataloader
        Args:
            data  : (list)
                0 : Input
                1 : Output
    """
    batch_size = len(data)

    x = torch.tensor([data[i][0] for i in range(batch_size)], dtype=torch.float)
    y = torch.tensor([data[i][1] for i in range(batch_size)], dtype=torch.float)

    return x, y

def collateE1(data):
    """
        Collate for E1 data
    """
    batch_size = len(data)

    x = torch.tensor([data[i][0][0] for i in range(batch_size) if data[i][0][0] is not None], dtype=torch.float)
    x_e1 = np.unique([data[i][0][1] for i in range(batch_size) if data[i][0][1] is not None])[0]
    y = torch.tensor([data[i][1] for i in range(batch_size) if data[i][1] is not None], dtype=torch.float)

    return (x, x_e1), y

def collateE1E2(data):
    """
        Collate for E1E2 data
    """
    batch_size = len(data)

    x = torch.tensor([data[i][0][0] for i in range(batch_size) if data[i][0][0] is not None], dtype=torch.float)
    x_e1 = np.unique([data[i][0][1] for i in range(batch_size) if data[i][0][1] is not None])[0]
    x_e2 = np.unique([data[i][0][2] for i in range(batch_size) if data[i][0][2] is not None])[0]
    y = torch.tensor([data[i][1] for i in range(batch_size) if data[i][1] is not None], dtype=torch.float)

    return (x, x_e1, x_e2), y

def collate(mode='r'):
    if isMode(mode, 'e1_e2'):
        return collateE1E2
    if isMode(mode, 'e1'):
        return collateE1
    elif isMode(mode, 'r'):
        return collateR

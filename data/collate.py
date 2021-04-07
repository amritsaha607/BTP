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

def collateE1E2Dom2(data):
    """
        Collate for E1E2 data on domain 2
    """
    batch_size = len(data)

    # print(data[0][0][1]['e1'][:5])
    eps = {key: torch.tensor(val.astype(str).astype(np.complex)) for key, val in data[0][0][1].items()} 
                                    # for a single batch, it belongs to a single class combination
                                    # so eps will be same for all of them, taking only 0th sample
    x = torch.tensor([data[i][0][0][0] for i in range(batch_size) if data[i][0][0][0] is not None], dtype=torch.float)
    x_e1 = np.unique([data[i][0][0][1] for i in range(batch_size) if data[i][0][0][1] is not None])[0]
    x_e2 = np.unique([data[i][0][0][2] for i in range(batch_size) if data[i][0][0][2] is not None])[0]
    y = torch.tensor([data[i][1] for i in range(batch_size) if data[i][1] is not None], dtype=torch.float)

    return ((x, x_e1, x_e2), eps), y

def collateE1E2E3(data):
    """
        Collate for E1E2E3 data
    """
    batch_size = len(data)

    x = torch.tensor([data[i][0][0] for i in range(batch_size) if data[i][0][0] is not None], dtype=torch.float)
    x_e1 = np.unique([data[i][0][1] for i in range(batch_size) if data[i][0][1] is not None])[0]
    x_e2 = np.unique([data[i][0][2] for i in range(batch_size) if data[i][0][2] is not None])[0]
    x_e3 = np.unique([data[i][0][3] for i in range(batch_size) if data[i][0][3] is not None])[0]
    y = torch.tensor([data[i][1] for i in range(batch_size) if data[i][1] is not None], dtype=torch.float)

    return (x, x_e1, x_e2, x_e3), y

def collate(mode='r', domain=0):
    if isMode(mode, 'e1_e2_e3'):
        return collateE1E2E3
    elif isMode(mode, 'e1_e2'):
        if domain==0:
            return collateE1E2
        elif domain==2:
            return collateE1E2Dom2
    elif isMode(mode, 'e1'):
        return collateE1
    elif isMode(mode, 'r'):
        return collateR

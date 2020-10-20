import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from data.utils import extractData
from utils.decorators import timer


class AreaDataset(Dataset):

    """
        Dataset that contains data of area distribution w.r.t wavelength 
        and the parameters used for the same calculation
    """

    def __init__(self, root='dataGeneration/data/', formats=['.csv']):
        super(AreaDataset, self).__init__()

        self.files = []
        for format_ in formats:
            self.files += glob.glob(os.path.join(root, '*', '*{}'.format(format_)))

        # print(self.files)
    
    def __getitem__(self, index):
        file = self.files[index]
        x, y = extractData(file)
        # x = torch.tensor(x)
        # y = torch.tensor(y)
        return x, y

    def __len__(self):
        return len(self.files)
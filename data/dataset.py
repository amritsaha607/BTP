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

    def __init__(self, root='dataGeneration/data/', formats=['.csv'], apply_factors=None):
        super(AreaDataset, self).__init__()

        self.files = []
        self.apply_factors = apply_factors
        for format_ in formats:
            self.files += glob.glob(os.path.join(root, '*', '*{}'.format(format_)))
    
    def __getitem__(self, index):
        file = self.files[index]
        x, y = extractData(
            file, 
            factors={
                'r': 1e9,
                'eps': 1,
                'lambd': 1e9,
                'A': 1e18
            } if self.apply_factors else None
        )
        return x, y

    def __len__(self):
        return len(self.files)
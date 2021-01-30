import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from data.utils import extractData
from utils.utils import isMode
from utils.decorators import timer


class AreaDataset(Dataset):

    """
        Dataset that contains data of area distribution w.r.t wavelength 
        and the parameters used for the same calculation
    """

    def __init__(self, 
        root='dataGeneration/data/', formats=['.csv'], factors=None, input_key='A_tot',
        mode='r',
        shuffle=True):
        super(AreaDataset, self).__init__()

        self.files = []
        self.factors = factors
        self.input_key = input_key
        self.mode = mode
        for format_ in formats:
            self.files += glob.glob(os.path.join(root, '*', '*{}'.format(format_)))
        if shuffle:
            random.shuffle(self.files)

        self.e1_materialCode = None
        if isMode(self.mode, 'e1'):
            self.e1_materialCode = {material.lower(): i for i, material in enumerate(os.listdir(os.path.join(root)))}

    def __getitem__(self, index):
        file = self.files[index]
        x, y = extractData(
            file, 
            input_key=self.input_key,
            mode=self.mode,
            factors=self.factors,
            e1_matCode=self.e1_materialCode
        )
        return x, y

    def __len__(self):
        return len(self.files)

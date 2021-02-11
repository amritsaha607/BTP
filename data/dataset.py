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
        shuffle=True,
        batch_size=None):

        if isMode(mode, 'e1') and batch_size==None:
            raise AssertionError("Please provide batch_size for mode {}".format(mode))

        super(AreaDataset, self).__init__()

        self.files = []
        self.factors = factors
        self.input_key = input_key
        self.mode = mode

        # Shuffling mode changed, data will be shuffled now 
        # but material wise data will be in sequential order
        for format_ in formats:
            for material in os.listdir(root):
                files = glob.glob(os.path.join(root, material, '*{}'.format(format_)))
                if shuffle:
                    random.shuffle(files)
                self.files += files

                # For e1 data, we'll have to add extra files (None) to fit it into batch_size
                # So that multiple material samples doesn't get into single batch
                if isMode(self.mode, 'e1') and (len(files) % batch_size != 0):
                    self.files += [None] * int(batch_size - len(files) % batch_size)

        self.e1_materialCode = None
        if isMode(self.mode, 'e1'):
            self.e1_materialCode = {material.lower(): i for i, material in enumerate(os.listdir(os.path.join(root)))}

    def __getitem__(self, index):
        file = self.files[index]

        if isMode(self.mode, 'e1') and file == None:
            return (None, None), None

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

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
        domain=0,
        shuffle=True,
        batch_size=None):

        if isMode(mode, 'e1') and batch_size==None:
            raise AssertionError("Please provide batch_size for mode {}".format(mode))

        super(AreaDataset, self).__init__()

        self.files = []
        self.factors = factors
        self.input_key = input_key
        self.mode = mode
        self.domain = domain

        if isMode(self.mode, 'e1_e2_e3'):
            for format_ in formats:
                for e1_mat in os.listdir(root):
                    e1_root = os.path.join(root, e1_mat)
                    for e2_mat in os.listdir(e1_root):
                        e2_root = os.path.join(e1_root, e2_mat)
                        for e3_mat in os.listdir(e2_root):
                            e3_root = os.path.join(e2_root, e3_mat)
                            files = glob.glob(os.path.join(e3_root, f"*{format_}"))
                            self.files += files

                            if shuffle:
                                random.shuffle(files)

                            if len(files) % batch_size != 0:
                                self.files += [None] * int(batch_size - len(files) % batch_size)

        elif isMode(self.mode, 'e1_e2'):
            for format_ in formats:
                for e1_mat in os.listdir(root):
                    e1_root = os.path.join(root, e1_mat)
                    for e2_mat in os.listdir(e1_root):
                        e2_root = os.path.join(e1_root, e2_mat)
                        files = glob.glob(os.path.join(e2_root, f"*{format_}"))
                        self.files += files

                        if shuffle:
                            random.shuffle(files)

                        if len(files) % batch_size != 0:
                            self.files += [None] * int(batch_size - len(files) % batch_size)

        else:
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

        self.setLambda()

        # self.e1_materialCode = None
        # if isMode(self.mode, 'e1'):
        #     self.e1_materialCode = {material.lower(): i for i, material in enumerate(os.listdir(os.path.join(root)))}

    def __getitem__(self, index):
        file = self.files[index]

        if file == None:
            y = None
            if isMode(self.mode, 'e1_e2_e3'):
                x = (None, None, None, None)
            elif isMode(self.mode, 'e1_e2'):
                x = (None, None, None)
            elif isMode(self.mode, 'e1'):
                x = (None, None)

            if self.domain == 2:
                x = (x, None)
            return x, y

        x, y = extractData(
            file, 
            input_key=self.input_key,
            mode=self.mode,
            domain=self.domain,
            factors=self.factors,
            # e1_matCode=self.e1_materialCode
        )
        return x, y

    def __len__(self):
        return len(self.files)

    def setLambda(self):
        file = self.files[0]
        self.lambd = torch.tensor(pd.read_csv(file)['lambd'].values)

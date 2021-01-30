"""
    Generates data of varying r1, r2 and e1
"""

import sys
sys.path.append("../../")

from data.utils import PermittivityCalculator
from dataGeneration.utils import *

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

random.seed(0)
np.random.seed(0)

materials = [
    "al2o3",
    "sio2",
]
pc = PermittivityCalculator()

f_name = '../../dataGeneration/csv/Au_interpolated_1.csv'
content = pd.read_csv(f_name)

wl = content["wl"].values / 1e3
DATA_ROOT = '../../dataGeneration/E1Data/'

lambd = 1e-9*content['wl'].values
e2 = content['er'].values + 1j*content['ei'].values
e3 = 1.00 + 1j*0.0


def helperUtil(data):
    r1, r2, e1, split, material, counter = data
    tr = {
        "r1": r1*1e-9,
        "r2": r2*1e-9,
    }
    te = {
        'e1': e1,
        'e2': e2,
        'e3': e3,
    }

    f_out = os.path.join(DATA_ROOT, split, material, f"{counter+1}.csv")

    getArea(
        tr, te, lambd,
        write=['A_abs'],
        f_out=f_out
    )

def helper(material, split="train", samples=None):

    """
        Args:
            samples : no of samples to generate for each material in validation set
    """

    print(f"Generating {material} data")
    counter = 0
    
    e1 = np.array([pc.getEps(wl_, element=material, mode="complex") for wl_ in wl])
    r1_min, r1_max = 5, 50
    
    if split=="train":
        for r1 in tqdm(range(r1_min, r1_max+1)):
            for r2 in range(r1+1, r1+101):
                data = (r1, r2, e1, split, material, counter)
                helperUtil(data)
                counter += 1

    elif split=="val":
        for counter in tqdm(range(samples)):
            r1 = random.random()*(r1_max-r1_min)
            r2 = r1 + (random.random()+0.01)*99
            data = (r1, r2, e1, split, material, counter)
            helperUtil(data)
            counter += 1

    else:
        raise ValueError("Unknown split {}".format(split))


for material in materials:
    helper(material, split="train")
    helper(material, split="val", samples=250)

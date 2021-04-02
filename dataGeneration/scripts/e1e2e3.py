"""
    Generates data of varying r1, r2, e1 & e2
"""

import sys
sys.path.append("../../")

from data.utils import PermittivityCalculator
from dataGeneration.utils import *
from utils.parameters import E1_CLASSES, E2_CLASSES, E3_CLASSES

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

random.seed(0)
np.random.seed(0)

pc = PermittivityCalculator()

f_name = '../../dataGeneration/csv/Au_interpolated_1.csv'
content = pd.read_csv(f_name)

wl = content["wl"].values / 1e3
DATA_ROOT = '../../dataGeneration/E1E2E3Data/'

lambd = 1e-9*content['wl'].values
e3 = 1.00 + 1j*0.0


def helperUtil(data):
    r1, r2, e1, e2, e3, split, e1_mat, e2_mat, e3_mat, counter = data
    tr = {
        "r1": r1*1e-9,
        "r2": r2*1e-9,
    }
    te = {
        'e1': e1,
        'e2': e2,
        'e3': e3,
    }

    f_out = os.path.join(DATA_ROOT, split, e1_mat.lower(), e2_mat.lower(), e3_mat.lower(), f"{counter+1}.csv")

    getArea(
        tr, te, lambd,
        write=['A_abs'],
        f_out=f_out
    )

def helper(e1_mat, e2_mat, e3_mat, split="train", samples=None):

    """
        Args:
            samples : no of samples to generate for each material in validation set
    """

    print(f"Generating e1 : {e1_mat}, e2 : {e2_mat} data, e3 : {e3_mat}")
    counter = 0
    
    e1 = np.array([pc.getEps(wl_, element=e1_mat, mode="complex") for wl_ in wl])
    e2 = pc.getEpsArr(element=e2_mat, interval=1, mode="complex", f_root="../")
    e3 = pc.getEpsVal(element=e3_mat)
    r1_min, r1_max = 5, 50
    
    if split=="train":
        for r1 in tqdm(range(r1_min, r1_max+1)):
            for r2 in range(r1+1, r1+101):
                data = (r1, r2, e1, e2, e3, split, e1_mat, e2_mat, e3_mat, counter)
                helperUtil(data)
                counter += 1

    elif split=="val":
        for counter in tqdm(range(samples)):
            r1 = random.random()*(r1_max-r1_min)
            r2 = r1 + (random.random()+0.01)*99
            data = (r1, r2, e1, e2, e3, split, e1_mat, e2_mat, e3_mat, counter)
            helperUtil(data)
            counter += 1

    else:
        raise ValueError("Unknown split {}".format(split))


E2_CLASSES = [
    'Au',
    'Ag',
]
for e1_material in tqdm(E1_CLASSES):
    for e2_material in E2_CLASSES:
        for e3_material in E3_CLASSES:
            helper(e1_material, e2_material, e3_material, split="train")
            helper(e1_material, e2_material, e3_material, split="val", samples=250)

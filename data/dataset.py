import numpy as np
from torch.utils.data import Dataset
import random


class AreaDataset(Dataset):

    """
        Dataset that contains data of area distribution w.r.t wavelength 
        and the parameters used for the same calculation
    """

    def __init__(self):
        super(AreaDataset, self).__init__()

    
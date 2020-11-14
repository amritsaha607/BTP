import os
import glob
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from dataGeneration.utils import getArea
from models import BasicModel

# Factors
factor_root = 'configs/data_factors/f2.yml'
factors = yaml.safe_load(open(factor_root))
factors = {key: float(val) for key, val in factors.items()}

# Data
f_name = 'dataGeneration/csv/Au_interpolated_5.csv'
content = pd.read_csv(f_name)

r_data = np.array([
    [1, 3],
    [1, 10],
    [1, 50],
    [1, 100],
    [10, 20],
    [10, 100],
    [10, 150],
    [20, 140],
    [30.77, 66.6],
    [45.89, 111.71],
    [42.58, 66.98]
])
r_data = r_data * 1e-9
lambd = 1e-9*content['wl'].values
eps = {
    'e1': np.sqrt(77/2.6),
    'e2': content['er'].values + 1j*content['ei'].values,
    'e3': 1.78,
}

# Model
n_samples = len(lambd)
model_out_dim = 2
model_ID = 10
model = BasicModel(
    input_dim = n_samples,
    out_dim = model_out_dim,
    model_id=model_ID,
)

print("Model {}".format(model_ID))

x = []
for [r1, r2] in r_data:
    r = {
        'r1': r1,
        'r2': r2,
    }
    A_sca, A_abs = getArea(r, eps, lambd)
    A_sca, A_abs = A_sca*factors['A'], A_abs*factors['A']
    x.append(A_abs)
    # print(A_abs.shape)

x = torch.tensor(np.array(x, dtype=np.float32))

# print(x.shape)

# Checkpoint at different versions
CHECKPOINT_DIR = 'checkpoints/MassData/{}'.format(model_ID)

for version in sorted(os.listdir(CHECKPOINT_DIR)):
    print(version)
    ckpt = os.path.join(CHECKPOINT_DIR, version)
    ckpt = glob.glob(os.path.join(ckpt, 'best*'))[0]
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(ckpt))
        x = x.cuda()
    else:
        model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
        x = x.detach().cpu()

    y = list(model(x).detach().cpu().numpy())
    # print(y.shape)

    for [r1, r2], [r1_pred, r2_pred] in zip(r_data, y):
        print('({:.2f}, {:.2f}) -> ({:.2f}, {:.2f})'.format(r1*1e9, r2*1e9, r1_pred*1e9/factors['r'], r2_pred*1e9/factors['r']))
    
    print()
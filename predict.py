from collections import defaultdict
import os
import glob
from tabulate import tabulate

from utils.parameters import E1_CLASSES
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wandb

from data.utils import PermittivityCalculator
from dataGeneration.utils import getArea
from models import BasicModel, E1Model
from utils.utils import isMode, getAreaE1Class

CUDA = torch.cuda.is_available()

# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=int, default=0, help='Model ID')
parser.add_argument('--data_factors', type=str, default='f2', 
    help='To spply factor in dataset')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r\
        r_e1 - r corresponding to each e1_Class"
)
parser.add_argument('--log', type=int, default=0, 
    help='To log the results in wandb or not')
args = parser.parse_args()


# Extract args
data_factors = args.data_factors
model_id = args.model
mode = args.mode
log = args.log


# Factors
DATA_FACTOR_ROOT = 'configs/data_factors'
data_factors = yaml.safe_load(open(os.path.join(DATA_FACTOR_ROOT, '{}.yml'.format(data_factors))))
data_factors = {key: float(val) for key, val in data_factors.items()}


# Data
if isMode(mode, 'e1'):
    f_name = 'dataGeneration/csv/Au_interpolated_1.csv'
    content = pd.read_csv(f_name)

    PREDICTION_FILE = "PredictionData/r_e1.csv"
    r_e1_data = pd.read_csv(PREDICTION_FILE)
    r_e1_data = {key: r_e1_data[key].values for key in r_e1_data.keys()}
    r_e1_data['r1'] = r_e1_data['r1'] * 1e-9
    r_e1_data['r2'] = r_e1_data['r2'] * 1e-9
    
    lambd = 1e-9*content['wl'].values
    e2 = content['er'].values + 1j*content['ei'].values

elif isMode(mode, 'r'):
    f_name = 'dataGeneration/csv/Au_interpolated_5.csv'
    content = pd.read_csv(f_name)

    PREDICTION_FILE = "PredictionData/r.csv"
    r_data = pd.read_csv(PREDICTION_FILE)
    r1s, r2s = r_data['r1'].values, r_data['r2'].values
    r_data = np.array([[r1s[i], r2s[i]] for i in range(len(r1s))])
    r_data = r_data * 1e-9
    lambd = 1e-9*content['wl'].values
    e2 = content['er'].values + 1j*content['ei'].values

    eps = {
        'e1': np.sqrt(77/2.6),
        'e2': e2,
        'e3': 1.78,
    }


# Model
n_samples = len(lambd)
model_out_dim = 2

if isMode(mode, 'e1'):
    model = E1Model(
        classes = E1_CLASSES,
        model_id = model_id,
        input_dim = n_samples,
        out_dim = model_out_dim,
    )
elif isMode(mode, 'r'):
    model = BasicModel(
        input_dim = n_samples,
        out_dim = model_out_dim,
        model_id = model_id,
    )

print("Model {}".format(model_id))


# Prepare Data for prediction
if isMode(mode, 'e1'):
    pc = PermittivityCalculator()
    e1_prediction_classes = list(set(r_e1_data['e1_cls']))

    r_e1_data = {
        e1_prediction_class: [
            [r1, r2] for (r1, r2, e1_cls) in zip(r_e1_data['r1'], r_e1_data['r2'], r_e1_data['e1_cls']) 
                if e1_cls==e1_prediction_class]
            for e1_prediction_class in e1_prediction_classes
    }

    r_e1_data_y, r_e1_data_x = {}, defaultdict(list)
    for e1_prediction_class in e1_prediction_classes:

        # Output is r1 & r2
        r_e1_data_y[e1_prediction_class] = torch.tensor(r_e1_data[e1_prediction_class])

        r_e1_data_x[e1_prediction_class] = getAreaE1Class(
            r_e1_data[e1_prediction_class],
            e1_prediction_class,
            data_file='dataGeneration/csv/Au_interpolated_1.csv',
            data_factors=data_factors,
            ret_mode='abs',
        )

    for key in r_e1_data_x:
        r_e1_data_x[key] = torch.tensor(r_e1_data_x[key]).type(torch.FloatTensor)

elif isMode(mode, 'r'):
    x = []
    for [r1, r2] in r_data:
        r = {
            'r1': r1,
            'r2': r2,
        }
        A_sca, A_abs = getArea(r, eps, lambd)
        A_sca, A_abs = A_sca*data_factors['A'], A_abs*data_factors['A']
        x.append(A_abs)

    x = torch.tensor(np.array(x, dtype=np.float32))

# Checkpoint at different versions
if isMode(mode, 'e1'):
    CHECKPOINT_DIR = f'checkpoints/{mode}/E1Data/{model_id}'
elif isMode(mode, 'r'):
    CHECKPOINT_DIR = f'checkpoints/{mode}/MassData/{model_id}'


# Logging
if log:
    if isMode(mode, 'e1'):
        WANDB_PROJECT_NAME = 'DL Nanophotonics'
        WANDB_PROJECT_DIR = '/content/wandb/'
        run_name = f"predict_{mode}_{model_id}"

        wandb.init(
            name=run_name, 
            project=WANDB_PROJECT_NAME, 
            dir=WANDB_PROJECT_DIR
        )

        config = wandb.config
        config.mode = mode
        config.model_id = model_id
        config.prediction_file = PREDICTION_FILE
        config.classes = E1_CLASSES


for version in sorted(os.listdir(CHECKPOINT_DIR)):
    print(version)
    ckpt = os.path.join(CHECKPOINT_DIR, version)
    ckpt = glob.glob(os.path.join(ckpt, 'best*'))[0]
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))

    if isMode(mode, 'e1'):

        wandb_table_cols = ['Original Class', 'r1', 'r2']
        wandb_table_rows = []

        for e1_cls in E1_CLASSES:
            wandb_table_cols += [
                f'r1_pred ({e1_cls})', 
                f'r2_pred ({e1_cls})', 
                f'err_pred ({e1_cls})',
            ]

        for e1_prediction_class in e1_prediction_classes:

            x = r_e1_data_x[e1_prediction_class]
            y = r_e1_data_y[e1_prediction_class]
            y_preds = {}
            err_preds = {}

            for e1_cls in E1_CLASSES:
                y_pred = model(x, e1_cls)

                # Reconstruct spectra from prediction using Maxwell's equations
                x_pred = getAreaE1Class(
                    list(y.numpy()),
                    e1_cls=e1_cls,
                    data_factors=data_factors,
                    ret_mode='abs',
                )
                x_pred = torch.tensor(x_pred)

                # Check the error between input spectra and reconstructed spectra
                err_spectra = np.abs(x - x_pred).sum()

                # Store prediction data
                y_preds[e1_cls] = y_pred
                err_preds[e1_cls] = err_spectra

            n_r = len(y)
            for i in range(n_r):
                wandb_table_row = [e1_prediction_class, y[i, 0]*1e9, y[i, 1]*1e9,]
                for e1_cls in E1_CLASSES:
                    wandb_table_row += [
                        y_preds[e1_cls][i, 0]*1e9/data_factors['r'],
                        y_preds[e1_cls][i, 1]*1e9/data_factors['r'],
                        err_preds[e1_cls]
                    ]
                wandb_table_rows.append(wandb_table_row)

            if log:
                table = wandb.Table(data=wandb_table_rows, columns=wandb_table_cols)
                wandb.log({f'table_{version}': table})

        print(tabulate(wandb_table_rows, headers=wandb_table_cols))

    elif isMode(mode, 'r'):
        if CUDA:
            model.cuda()
            x = x.cuda()
        else:
            x = x.detach().cpu()

        y = list(model(x).detach().cpu().numpy())

        for [r1, r2], [r1_pred, r2_pred] in zip(r_data, y):
            print('({:.2f}, {:.2f}) -> ({:.2f}, {:.2f})'.format(
                r1*1e9, r2*1e9, r1_pred*1e9/data_factors['r'], r2_pred*1e9/data_factors['r']))

    print()

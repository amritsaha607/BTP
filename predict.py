from collections import defaultdict
from models.model import E1E2Model
import os
import glob

from utils.vis import plotArrMulti
from tabulate import tabulate

from utils.parameters import E1_CLASSES, E2_CLASSES
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
from utils.utils import excelDfWriter, getAreaE1E2Class, isMode, getAreaE1Class

CUDA = torch.cuda.is_available()

# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--domain', type=int, default=0, 
    help="Pipeline domain\
        0 -> model predicts (r1, r2)\
        1 -> model predicts (r1/r2, r2-r1)")
parser.add_argument('--model', type=int, default=0, help='Model ID')
parser.add_argument('--data_factors', type=str, default='f2', 
    help='To spply factor in dataset')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r\
        r_e1 - predict r with e1 class\
        r_e1_e2 - predict r with e1 & e2 classes"
)
parser.add_argument('--log', type=int, default=0, 
    help='To log the results in wandb or not')
parser.add_argument('--plot', type=int, default=0, 
    help='To log the plots in wandb or not')
args = parser.parse_args()


# Extract args
domain = args.domain
data_factors = args.data_factors
model_id = args.model
mode = args.mode
log = args.log
plot = args.plot


EXPORT_EXCEL_FILENAME = f'PredictionData/Results/dom{domain}/{mode}/model_{model_id}.xlsx'
dataframes = []

if not os.path.exists(os.path.dirname(EXPORT_EXCEL_FILENAME)):
    os.makedirs(os.path.dirname(EXPORT_EXCEL_FILENAME))


# Factors
DATA_FACTOR_ROOT = 'configs/data_factors'
data_factors = yaml.safe_load(open(os.path.join(DATA_FACTOR_ROOT, '{}.yml'.format(data_factors))))
data_factors = {key: float(val) for key, val in data_factors.items()}


# Extract data_factors
fr = data_factors['r']
fA = data_factors['A']
f_eps = data_factors['eps']
f_lambd = data_factors['lambd']


# Data
if isMode(mode, 'e1_e2'):
    f_name = 'dataGeneration/csv/Au_interpolated_1.csv'
    content = pd.read_csv(f_name)

    PREDICTION_FILE = "PredictionData/r_e1_e2.csv"
    r_e1_e2_data = pd.read_csv(PREDICTION_FILE)
    r_e1_e2_data = {key: r_e1_e2_data[key].values for key in r_e1_e2_data.keys()}
    r_e1_e2_data['r1'] = r_e1_e2_data['r1'] * 1e-9
    r_e1_e2_data['r2'] = r_e1_e2_data['r2'] * 1e-9
    
    lambd = 1e-9*content['wl'].values
    e1e2_classes = [f'{e1_cls},{e2_cls}' for e1_cls in E1_CLASSES for e2_cls in E2_CLASSES]
    # e2 = content['er'].values + 1j*content['ei'].values

elif isMode(mode, 'e1'):
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

if isMode(mode, 'e1_e2'):
    model = E1E2Model(
        e1_classes = E1_CLASSES,
        e2_classes = E2_CLASSES,
        model_id = model_id,
        input_dim = n_samples,
        out_dim = model_out_dim,
    )
elif isMode(mode, 'e1'):
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
if isMode(mode, 'e1_e2'):
    pc = PermittivityCalculator()

    e1_classes, e2_classes = list(set(r_e1_e2_data['e1_cls'])), list(set(r_e1_e2_data['e2_cls']))
    e1e2_prediction_classes = [f'{e1_cls},{e2_cls}' for e1_cls in e1_classes for e2_cls in e2_classes]

    r_e1_e2_data = {
        e1e2_prediction_class: [
            [r1, r2] for (r1, r2, e1_cls, e2_cls) in zip(r_e1_e2_data['r1'], r_e1_e2_data['r2'], r_e1_e2_data['e1_cls'], r_e1_e2_data['e2_cls'])
                if f'{e1_cls},{e2_cls}'==e1e2_prediction_class]
        for e1e2_prediction_class in e1e2_prediction_classes
    }

    r_e1_e2_data_y, r_e1_e2_data_x = {}, defaultdict(list)
    for e1e2_prediction_class in e1e2_prediction_classes:

        # Output is r1 & r2
        r_e1_e2_data_y[e1e2_prediction_class] = torch.tensor(r_e1_e2_data[e1e2_prediction_class])

        r_e1_e2_data_x[e1e2_prediction_class] = getAreaE1E2Class(
            r_e1_e2_data[e1e2_prediction_class],
            e1e2_prediction_class.split(',')[0],
            e1e2_prediction_class.split(',')[1],
            data_file='dataGeneration/csv/Au_interpolated_1.csv',
            data_factors=data_factors,
            ret_mode='abs',
        )

    for key in r_e1_e2_data_x:
        r_e1_e2_data_x[key] = torch.tensor(r_e1_e2_data_x[key]).type(torch.FloatTensor)

elif isMode(mode, 'e1'):
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
if isMode(mode, 'e1_e2'):
    CHECKPOINT_DIR = f'checkpoints/domain_{domain}/{mode}/E1E2Data/{model_id}'
elif isMode(mode, 'e1'):
    CHECKPOINT_DIR = f'checkpoints/domain_{domain}/{mode}/E1Data/{model_id}'
elif isMode(mode, 'r'):
    CHECKPOINT_DIR = f'checkpoints/domain_{domain}/{mode}/MassData/{model_id}'


for version in sorted(os.listdir(CHECKPOINT_DIR)):
    print(version)
    ckpt = os.path.join(CHECKPOINT_DIR, version)
    ckpt = glob.glob(os.path.join(ckpt, 'best*'))[0]
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
    
    if isMode(mode, 'e1_e2'):

        wandb_table_cols = ['Original Class', 'r1', 'r2']
        if plot:
            wandb_table_cols.append("Plots")
        wandb_table_rows = []

        for e1e2_cls in e1e2_classes:
            wandb_table_cols += [
                f'r1_pred ({e1e2_cls})', 
                f'r2_pred ({e1e2_cls})', 
                f'err_pred ({e1e2_cls})',
            ]

        for e1e2_prediction_class in e1e2_prediction_classes:

            x = r_e1_e2_data_x[e1e2_prediction_class]
            y = r_e1_e2_data_y[e1e2_prediction_class]
            x_preds = {}
            y_preds = {}
            err_preds = {}

            for e1e2_cls in e1e2_classes:
                [e1_cls, e2_cls] = e1e2_cls.split(',')
                y_pred = model(x, e1_cls, e2_cls).detach()

                # Reconstruct spectra from prediction using Maxwell's equations
                x_pred = getAreaE1E2Class(
                    list(y_pred.numpy() / fr),
                    e1_cls=e1_cls,
                    e2_cls=e2_cls,
                    data_file='dataGeneration/csv/Au_interpolated_1.csv',
                    data_factors=data_factors,
                    ret_mode='abs',
                )
                x_pred = torch.tensor(x_pred)

                # Check the error between input spectra and reconstructed spectra
                err_spectra = np.abs(x - x_pred).sum(axis=1)

                # Store prediction data
                x_preds[e1e2_cls] = x_pred
                y_preds[e1e2_cls] = y_pred
                err_preds[e1e2_cls] = err_spectra

            n_r = len(y)
            for i in range(n_r):
                wandb_table_row = [e1e2_prediction_class, y[i, 0]*1e9, y[i, 1]*1e9]
                
                # Make the plots
                if plot:
                    fig = plotArrMulti(
                        [x[i].numpy()] + [x_preds[key][i].numpy() for key in sorted(x_preds.keys())],
                        labels = [f'GT ({e1e2_prediction_class})'] + [key for key in sorted(x_preds.keys())],
                        debug = False,
                        ret_mode = 'fig',
                        continuous = True,
                        size_=(4, 4),
                    )
                    wandb_table_row.append(fig)
                
                # Insert predictions and corresponding errors
                for e1e2_cls in e1e2_classes:
                    wandb_table_row += [
                        y_preds[e1e2_cls][i, 0]*1e9/data_factors['r'],
                        y_preds[e1e2_cls][i, 1]*1e9/data_factors['r'],
                        err_preds[e1e2_cls][i],
                    ]
                
                wandb_table_row = [float(elem) if torch.is_tensor(elem) else elem 
                    for elem in wandb_table_row]
                wandb_table_rows.append(wandb_table_row)

        if log:
            df = pd.DataFrame(
                data=wandb_table_rows,
                columns=wandb_table_cols,
                index=['']*len(wandb_table_rows),
            )
            dataframes.append([df, version])

        print(tabulate(wandb_table_rows, headers=wandb_table_cols))

    elif isMode(mode, 'e1'):

        wandb_table_cols = ['Original Class', 'r1', 'r2']
        if plot:
            wandb_table_cols.append("Plots")
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
            x_preds = {}
            y_preds = {}
            err_preds = {}

            for e1_cls in E1_CLASSES:
                y_pred = model(x, e1_cls).detach()

                # Reconstruct spectra from prediction using Maxwell's equations
                x_pred = getAreaE1Class(
                    list(y_pred.numpy() / fr),
                    e1_cls=e1_cls,
                    data_file='dataGeneration/csv/Au_interpolated_1.csv',
                    data_factors=data_factors,
                    ret_mode='abs',
                )
                x_pred = torch.tensor(x_pred)

                # Check the error between input spectra and reconstructed spectra
                err_spectra = np.abs(x - x_pred).sum(axis=1)

                # Store prediction data
                x_preds[e1_cls] = x_pred
                y_preds[e1_cls] = y_pred
                err_preds[e1_cls] = err_spectra

            n_r = len(y)
            for i in range(n_r):
                wandb_table_row = [e1_prediction_class, y[i, 0]*1e9, y[i, 1]*1e9]
                
                # Make the plots
                if plot:
                    fig = plotArrMulti(
                        [x[i].numpy()] + [x_preds[key][i].numpy() for key in sorted(x_preds.keys())],
                        labels = [f'GT ({e1_prediction_class})'] + [key for key in sorted(x_preds.keys())],
                        debug = False,
                        ret_mode = 'fig',
                        continuous = True,
                        size_=(2, 2),
                    )
                    wandb_table_row.append(fig)
                
                # Insert predictions and corresponding errors
                for e1_cls in E1_CLASSES:
                    wandb_table_row += [
                        y_preds[e1_cls][i, 0]*1e9/data_factors['r'],
                        y_preds[e1_cls][i, 1]*1e9/data_factors['r'],
                        err_preds[e1_cls][i],
                    ]
                
                wandb_table_row = [float(elem) if torch.is_tensor(elem) else elem 
                    for elem in wandb_table_row]
                wandb_table_rows.append(wandb_table_row)

        if log:
            df = pd.DataFrame(
                data=wandb_table_rows,
                columns=wandb_table_cols,
                index=['']*len(wandb_table_rows),
            )
            dataframes.append([df, version])

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


# Logging
if log:
    if isMode(mode, 'e1'):
        WANDB_PROJECT_NAME = 'DL Nanophotonics'
        WANDB_PROJECT_DIR = '/content/wandb/'
        run_name = f"predict_{mode}_{model_id}_dom{domain}"

        wandb.init(
            name=run_name, 
            project=WANDB_PROJECT_NAME, 
            dir=WANDB_PROJECT_DIR
        )

        config = wandb.config
        config.mode = mode
        config.model_ID = model_id
        config.prediction_file = PREDICTION_FILE
        config.e1_classes = E1_CLASSES
        if isMode(mode, 'e2'):
            config.e2_classes = E2_CLASSES


# Log in wandb and export to xlsx
if isMode(mode, 'e1') or isMode(mode, 'e2'):

    dfs = [elem[0] for elem in dataframes]
    sheets = [elem[1] for elem in dataframes]
    excelDfWriter(dfs,
                  filename=EXPORT_EXCEL_FILENAME,
                  sheet_name=sheets,
                  dispose=True
                  )

    if log:
        for [df, sheet_name] in dataframes:
            df.drop('Plots', 1, inplace=True)
            table = wandb.Table(data=list(df.values),
                                columns=list(df.keys()))
            wandb.log({f'table_{sheet_name}': table})

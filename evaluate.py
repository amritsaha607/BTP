import yaml
import os
import glob
import argparse
import wandb
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import BasicModel
from data.dataset import AreaDataset
from data.collate import collate
from utils.utils import getLabel
from utils.decorators import timer
from utils.operations import dictAdd, dictMultiply
from eval.utils import evaluate


# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=str, default='v0', help='Version of model to evaluate')
parser.add_argument('--verbose', type=int, default=1, help='To show evaluation progress or not')
parser.add_argument('--data_factors', type=str, default='f0', 
    help='To spply factor in dataset')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r"
)
args = parser.parse_args()

version = args.version
verbose = args.verbose
data_factors = args.data_factors
mode = args.mode
cfg_path = os.path.join('configs/{}.yml'.format(version.replace('_', '/')))
configs = yaml.safe_load(open(cfg_path))

random_seed = int(configs['random_seed'])
batch_size = int(configs['batch_size'])
data_root = configs['val_root']
CHECKPOINT_DIR = configs['CHECKPOINT_DIR']
ckpt_dir = os.path.join('checkpoints', version.replace('_', '/'))
ckpt = glob.glob(os.path.join(ckpt_dir, 'best*.pth'))
if len(ckpt)==0:
    raise ValueError("No checkpoint found in location {}".format(os.path.join(ckpt_dir, 'best*.pth')))
ckpt = ckpt[0]

# Set random seeds
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# Dataset
DATA_FACTOR_ROOT = 'configs/data_factors'
data_factors = yaml.safe_load(open(os.path.join(DATA_FACTOR_ROOT, '{}.yml'.format(data_factors))))
data_factors = {key: float(val) for key, val in data_factors.items()}
dataset = AreaDataset(
    root=data_root,
    formats=['.csv'],
    factors=data_factors,
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)

# Samples
f = glob.glob(os.path.join(data_root, '*', '*.csv'))[0]
n_samples = pd.read_csv(f).values.shape[0]

# Model
if mode=='default':
    model_out_dim = 6+2*n_samples
elif mode=='r':
    model_out_dim = 2

model = BasicModel(
    input_dim = n_samples,
    out_dim = model_out_dim
)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(ckpt))


run_name = "eval_{}".format(version)
wandb.init(name=run_name, project="DL Nanophotonics", dir='/content/wandb/')

config = wandb.config

config.version = version
config.batch_size = batch_size
config.data_factors = args.data_factors
config.CHECKPOINT_DIR = CHECKPOINT_DIR
config.cuda = torch.cuda.is_available()
config.log_interval = 1

loggs = evaluate(
    model,
    dataloader,
    mode=mode,
    verbose=verbose,
    err_acc_meters=[1, 5, 10, 20, 50, 100]
)
for logg in loggs:
    wandb.log(logg)
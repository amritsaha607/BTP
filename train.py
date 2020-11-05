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
from models.utils.loss import SeperateLoss
from data.dataset import AreaDataset
from data.collate import collate
from utils.utils import getLossWeights, getLabel
from utils.decorators import timer
from utils.operations import dictAdd, dictMultiply


# Add argument via parser
parser = argparse.ArgumentParser()

parser.add_argument('--version', type=str, default='v0', help='Version of experiment')
parser.add_argument('--verbose', type=int, default=1, help='To show training progress or not')
parser.add_argument('--data_factors', type=str, default='f0', 
    help='To spply factor in dataset')
parser.add_argument(
    '--mode', type=str, default='default', 
    help="Mode selects which parameter to predict\
        default - predict all\
        r - predict r"
)
parser.add_argument('--save', type=int, default=100, help='Version of experiment')
args = parser.parse_args()


version = args.version
verbose = args.verbose
data_factors = args.data_factors
mode = args.mode
save = args.save
cfg_path = os.path.join('configs/{}.yml'.format(version.replace('_', '/')))
configs = yaml.safe_load(open(cfg_path))

random_seed = int(configs['random_seed'])
batch_size = int(configs['batch_size'])
n_epoch = int(configs['n_epoch'])
train_root = configs['train_root']
val_root = configs['val_root']
optimizer_ = configs['optimizer']
learning_rate = float(configs['lr'])
weight_decay = float(configs['weight_decay'])
adam_eps = float(configs['adam_eps'])
adam_amsgrad = bool(configs['adam_amsgrad'])
CHECKPOINT_DIR = configs['CHECKPOINT_DIR']
ckpt_dir = os.path.join('checkpoints', version.replace('_', '/'))
LOSS_WEIGHT_DIR = configs['LOSS_WEIGHT_DIR']
input_key = configs['input_key'] if 'input_key' in configs else 'A_tot'

# Checkpoint Directory
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Set seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Dataset
DATA_FACTOR_ROOT = 'configs/data_factors'
data_factors = yaml.safe_load(open(os.path.join(DATA_FACTOR_ROOT, '{}.yml'.format(data_factors))))
data_factors = {key: float(val) for key, val in data_factors.items()}
train_set = AreaDataset(
    root=train_root,
    formats=['.csv'],
    factors=data_factors,
    input_key=input_key,
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)
val_set = AreaDataset(
    root=val_root,
    formats=['.csv'],
    factors=data_factors,
    input_key=input_key,
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    num_workers=8,
    collate_fn=collate,
    drop_last=False,
)


# Samples
f = glob.glob(os.path.join(train_root, '*', '*.csv'))[0]
n_samples = pd.read_csv(f).values.shape[0]

# Model, loss
if mode=='default':
    model_out_dim = 6+2*n_samples
elif mode=='r':
    model_out_dim = 2

model = BasicModel(
    input_dim = n_samples,
    out_dim = model_out_dim
)
criterion = SeperateLoss()
loss_mode, loss_split_mode = 'mse', 'split_re'
loss_weights = None
if 'loss_weight' in configs:
    loss_mode = 'weighted'
    loss_split_mode += '_weighted'
    loss_weight_cfg = os.path.join(LOSS_WEIGHT_DIR, '{}.yml'.format(configs['loss_weight']))
    loss_weight_cfg = yaml.safe_load(open(loss_weight_cfg))
    loss_weight_cfg = {key: float(val) for key, val in loss_weight_cfg.items()}
    loss_weights = getLossWeights(
        weights_dict=loss_weight_cfg,
        n=n_samples
    )

if torch.cuda.is_available():
    model.cuda()
    criterion.cuda()

# Training Function
@timer
def train(epoch, loader, optimizer, metrics=[], 
            verbose=1, topups=['loss_split_re']):

    """
        epoch : Epoch no
        loader : Training dataloader
        optimizer : Optimizer used
        metrics : metrics to log
    """

    if (mode=='r' or mode=='eps') and ('loss_split_re' in topups):
        raise ValueError("You can't add {} topup in {} mode"
            .format("loss_split_re", mode))

    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    if 'loss_split_re' in topups:
        tot_loss_split = None

    model.train()
    for batch_idx, (x, y) in enumerate(loader):

        y = getLabel(y, mode=mode)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = criterion(y_pred, y, mode=loss_mode, run='train', weights=loss_weights)
        loss.backward()
        optimizer.step()

        if 'loss_split_re' in topups:
            loss_split = criterion(
                y_pred, y, 
                mode=loss_split_mode, run='train', 
                weights=loss_weights
            )
            tot_loss_split = dictAdd([tot_loss_split, loss_split]) if tot_loss_split else loss_split

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Training - [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
            )
            print(progress, end='\r')

    print()
    logg = {
        'training_loss': tot_loss/loss_count,
    }

    if 'loss_split_re' in topups:
        for key in tot_loss_split:
            tot_loss_split[key] /= loss_count
        logg.update(tot_loss_split)

    return logg


# Validation Function
@timer
def validate(epoch, loader, metrics=[], 
            verbose=1, topups=['loss_split_re']):

    """
        epoch : Epoch no
        loader : Validation dataloader
        metrics : metrics to log
    """

    n = len(loader)
    tot_loss, loss_count = 0.0, 0
    if 'loss_split_re' in topups:
        tot_loss_split = None

    model.eval()
    for batch_idx, (x, y) in enumerate(loader):

        y = getLabel(y, mode=mode)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = criterion(y_pred, y, mode=loss_mode, run='val', weights=loss_weights)

        if 'loss_split_re' in topups:
            loss_split = criterion(
                y_pred, y, 
                mode=loss_split_mode, run='val',
                weights=loss_weights
            )
            tot_loss_split = dictAdd([tot_loss_split, loss_split]) if tot_loss_split else loss_split

        if not math.isnan(loss.item()):
            tot_loss += loss.item()
            loss_count += 1

        if verbose:
            n_arrow = 50*(batch_idx+1)//n
            progress = "Validation - [{}>{}] ({}/{}) loss : {:.4f}, avg_loss : {:.4f}".format(
                "="*n_arrow, "-"*(50-n_arrow), (batch_idx+1), n, loss.item(), tot_loss/loss_count
            )
            print(progress, end='\r')

    print()
    logg = {
        'val_loss': tot_loss/loss_count,
    }

    if 'loss_split_re' in topups:
        for key in tot_loss_split:
            tot_loss_split[key] /= loss_count
        logg.update(tot_loss_split)

    return logg



def run():

    # Optimizer
    if optimizer_=='adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=adam_eps,
            amsgrad=adam_amsgrad
        )

    # Scheduler (optional)
    scheduler = None
    if 'scheduler' in configs:
        sch_factor = configs['scheduler']
        lr_lambda = lambda epoch: sch_factor**epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize wandb
    run_name = "train_{}".format(version)
    wandb.init(name=run_name, project="DL Nanophotonics", dir='/content/wandb/')
    wandb.watch(model, log='all')

    config = wandb.config

    config.version = version
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.train_root = train_root
    config.val_root = val_root
    config.data_factors = args.data_factors
    config.optimizer = optimizer_
    config.lr = learning_rate
    config.weight_decay = weight_decay
    config.adam_eps = adam_eps
    config.amsgrad = adam_amsgrad
    config.CHECKPOINT_DIR = CHECKPOINT_DIR
    config.scheduler = sch_factor if scheduler is not None else None
    config.cuda = torch.cuda.is_available()
    config.log_interval = 1
    
    BEST_LOSS = float('inf')
    topups = []
    # topups = ['loss_split_re']

    # Train & Validate over multiple epochs
    for epoch in range(1, n_epoch+1):

        print("Epoch {}".format(epoch))

        logg = {}
        
        logg_train = train(
            epoch,
            train_loader,
            optimizer,
            metrics=[],
            topups=topups,
            verbose=verbose
        )
        logg_val = validate(
            epoch,
            val_loader,
            metrics=[],
            topups=topups,
            verbose=verbose
        )

        logg.update(logg_train)
        logg.update(logg_val)

        if scheduler and epoch%10==0:
            scheduler.step()
            print("\nepoch {}, lr : {}\n"
                    .format(epoch, [param_group['lr'] for param_group in optimizer.param_groups]))

        wandb.log(logg, step=epoch)

        if save:
            if logg['training_loss'] < BEST_LOSS:
                BEST_LOSS = logg['training_loss']
                os.system('rm {}'.format(os.path.join(ckpt_dir, 'best_*.pth')))
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_{}.pth'.format(epoch)))
            if epoch==n_epoch:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'latest_{}.pth'.format(epoch)))


if __name__=='__main__':
    run()